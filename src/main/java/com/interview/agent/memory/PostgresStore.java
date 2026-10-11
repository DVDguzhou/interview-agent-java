package com.interview.agent.memory;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * PostgreSQL 持久化层：用户长期画像 + 面试记录写入。
 * <p>
 * 表结构见 {@code resources/db/schema.sql}（启动时由 spring.sql.init 幂等执行）。
 * 画像中的技能等级 / 薄弱点、面试评估报告 / 复习计划均以 JSONB 存储，配合 GIN 索引支撑多维查询，
 * 查询侧见 {@link com.interview.agent.record.InterviewRecordRepository}。
 */
@Slf4j
@Component
public class PostgresStore {

    private final NamedParameterJdbcTemplate jdbc;
    private final ObjectMapper objectMapper = new ObjectMapper()
            .registerModule(new JavaTimeModule())
            .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);

    public PostgresStore(NamedParameterJdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public void saveProfile(UserProfile profile) {
        try {
            jdbc.update("""
                    INSERT INTO user_profiles (user_id, name, skill_level, weak_points)
                    VALUES (:userId, :name, CAST(:skillLevel AS jsonb), CAST(:weakPoints AS jsonb))
                    ON CONFLICT (user_id) DO UPDATE SET
                        name        = EXCLUDED.name,
                        skill_level = EXCLUDED.skill_level,
                        weak_points = EXCLUDED.weak_points,
                        updated_at  = now()
                    """, new MapSqlParameterSource()
                    .addValue("userId", profile.getUserId())
                    .addValue("name", profile.getName() != null ? profile.getName() : "")
                    .addValue("skillLevel", toJson(profile.getSkillLevel(), "{}"))
                    .addValue("weakPoints", toJson(profile.getWeakPoints(), "[]")));
        } catch (Exception e) {
            log.error("[PostgresStore] 保存 Profile 失败: {}", e.getMessage());
        }
    }

    public UserProfile loadProfile(String userId) {
        try {
            List<UserProfile> rows = jdbc.query(
                    "SELECT user_id, name, skill_level::text AS skill_level, weak_points::text AS weak_points "
                            + "FROM user_profiles WHERE user_id = :userId",
                    Map.of("userId", userId),
                    (rs, i) -> {
                        UserProfile p = new UserProfile();
                        p.setUserId(rs.getString("user_id"));
                        p.setName(rs.getString("name"));
                        try {
                            p.setSkillLevel(objectMapper.readValue(rs.getString("skill_level"),
                                    new TypeReference<Map<String, String>>() {}));
                            p.setWeakPoints(objectMapper.readValue(rs.getString("weak_points"),
                                    new TypeReference<List<UserProfile.WeakPoint>>() {}));
                        } catch (Exception e) {
                            throw new IllegalStateException(e);
                        }
                        return p;
                    });
            return rows.isEmpty() ? null : rows.get(0);
        } catch (Exception e) {
            log.error("[PostgresStore] 加载 Profile 失败: {}", e.getMessage());
            return null;
        }
    }

    /**
     * 写入面试记录（按 session_id 幂等）。report / reviewPlan 为任意可序列化对象，落库为 JSONB。
     */
    public void saveInterviewRecord(String userId, UserProfile.InterviewRecord record, String overallLevel,
                                    Object report, Object reviewPlan) {
        try {
            jdbc.update("""
                    INSERT INTO interview_records
                        (user_id, session_id, position, overall_score, overall_level, report_json, review_plan_json)
                    VALUES (:userId, :sessionId, :position, :score, :level, CAST(:report AS jsonb), CAST(:plan AS jsonb))
                    ON CONFLICT (session_id) DO UPDATE SET
                        overall_score    = EXCLUDED.overall_score,
                        overall_level    = EXCLUDED.overall_level,
                        report_json      = EXCLUDED.report_json,
                        review_plan_json = EXCLUDED.review_plan_json
                    """, new MapSqlParameterSource()
                    .addValue("userId", userId)
                    .addValue("sessionId", record.getSessionId())
                    .addValue("position", record.getPosition() != null ? record.getPosition() : "")
                    .addValue("score", record.getOverallScore())
                    .addValue("level", overallLevel != null ? overallLevel : "")
                    .addValue("report", toJson(report, "{}"))
                    .addValue("plan", toJson(reviewPlan, "{}")));
        } catch (Exception e) {
            log.error("[PostgresStore] 保存面试记录失败: {}", e.getMessage());
        }
    }

    private String toJson(Object value, String fallback) {
        if (value == null) return fallback;
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            return fallback;
        }
    }
}
