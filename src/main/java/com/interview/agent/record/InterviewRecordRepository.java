package com.interview.agent.record;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Repository;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * 面试记录查询（只读）。
 * <p>
 * 结构化评估结果存在 {@code interview_records.report_json}（JSONB），评级 / 薄弱点 / 遗漏知识点
 * 等条件统一转换成 {@code @>} 包含查询，命中 {@code idx_interview_records_report} GIN 索引；
 * 维度分数区间用 {@code ->>} 表达式过滤。
 */
@Repository
public class InterviewRecordRepository {

    private static final String SUMMARY_COLUMNS = """
            session_id, position, overall_score, overall_level, created_at,
            COALESCE(report_json -> 'dimension_score', '{}'::jsonb)::text AS dimension_score,
            COALESCE(report_json -> 'weaknesses', '[]'::jsonb)::text     AS weaknesses
            """;

    private final NamedParameterJdbcTemplate jdbc;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public InterviewRecordRepository(NamedParameterJdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public List<InterviewRecordSummary> search(String userId, InterviewRecordQuery q) {
        StringBuilder sql = new StringBuilder("SELECT ").append(SUMMARY_COLUMNS)
                .append(" FROM interview_records WHERE user_id = :userId");
        MapSqlParameterSource params = new MapSqlParameterSource("userId", userId);

        if (notBlank(q.position())) {
            sql.append(" AND position ILIKE :position");
            params.addValue("position", "%" + q.position().trim() + "%");
        }
        if (q.minScore() != null) {
            sql.append(" AND overall_score >= :minScore");
            params.addValue("minScore", q.minScore());
        }
        if (q.maxScore() != null) {
            sql.append(" AND overall_score <= :maxScore");
            params.addValue("maxScore", q.maxScore());
        }
        if (notBlank(q.level())) {
            sql.append(" AND report_json @> jsonb_build_object('overall_level', CAST(:level AS text))");
            params.addValue("level", q.level().trim().toUpperCase());
        }
        if (notBlank(q.weakness())) {
            sql.append(" AND report_json @> jsonb_build_object('weaknesses', jsonb_build_array(CAST(:weakness AS text)))");
            params.addValue("weakness", q.weakness().trim());
        }
        if (notBlank(q.missedKeyPoint())) {
            sql.append(" AND report_json @> jsonb_build_object('detailed_review', jsonb_build_array(")
                    .append("jsonb_build_object('key_points_missed', jsonb_build_array(CAST(:missed AS text)))))");
            params.addValue("missed", q.missedKeyPoint().trim());
        }
        if (notBlank(q.dimension()) && q.minDimensionScore() != null) {
            sql.append(" AND (report_json -> 'dimension_score' ->> CAST(:dimension AS text))::float8 >= :minDim");
            params.addValue("dimension", q.dimension().trim());
            params.addValue("minDim", q.minDimensionScore());
        }

        sql.append(" ORDER BY created_at DESC LIMIT :limit OFFSET :offset");
        params.addValue("limit", q.effectiveLimit());
        params.addValue("offset", q.effectiveOffset());

        return jdbc.query(sql.toString(), params, summaryMapper(false));
    }

    public Optional<InterviewRecordSummary> findBySession(String userId, String sessionId) {
        List<InterviewRecordSummary> rows = jdbc.query(
                "SELECT " + SUMMARY_COLUMNS + ", report_json::text AS report, review_plan_json::text AS review_plan "
                        + "FROM interview_records WHERE user_id = :userId AND session_id = :sessionId",
                Map.of("userId", userId, "sessionId", sessionId),
                summaryMapper(true));
        return rows.stream().findFirst();
    }

    /** 薄弱点频次统计：展开所有报告的 weaknesses 数组聚合计数 */
    public List<Map<String, Object>> weaknessStats(String userId, int limit) {
        return jdbc.queryForList("""
                SELECT w AS topic, COUNT(*) AS count
                FROM interview_records r,
                     LATERAL jsonb_array_elements_text(COALESCE(r.report_json -> 'weaknesses', '[]'::jsonb)) AS w
                WHERE r.user_id = :userId
                GROUP BY w
                ORDER BY count DESC, topic
                LIMIT :limit
                """, new MapSqlParameterSource("userId", userId).addValue("limit", Math.max(1, Math.min(limit, 50))));
    }

    private RowMapper<InterviewRecordSummary> summaryMapper(boolean withDetail) {
        return (rs, i) -> new InterviewRecordSummary(
                rs.getString("session_id"),
                rs.getString("position"),
                rs.getDouble("overall_score"),
                rs.getString("overall_level"),
                read(rs.getString("dimension_score"), new TypeReference<Map<String, Double>>() {}),
                read(rs.getString("weaknesses"), new TypeReference<List<String>>() {}),
                rs.getObject("created_at", OffsetDateTime.class),
                withDetail ? readTree(rs.getString("report")) : null,
                withDetail ? readTree(rs.getString("review_plan")) : null);
    }

    private <T> T read(String json, TypeReference<T> type) {
        try {
            return json == null ? null : objectMapper.readValue(json, type);
        } catch (Exception e) {
            return null;
        }
    }

    private JsonNode readTree(String json) {
        try {
            return json == null ? null : objectMapper.readTree(json);
        } catch (Exception e) {
            return null;
        }
    }

    private static boolean notBlank(String s) {
        return s != null && !s.isBlank();
    }
}
