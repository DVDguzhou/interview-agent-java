package com.interview.agent.record;

import com.interview.agent.memory.PostgresStore;
import com.interview.agent.memory.UserProfile;
import com.interview.agent.model.EvaluationReport;
import com.interview.agent.rag.QuestionBankStore;
import com.interview.agent.rag.RagDocument;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.jdbc.datasource.init.ResourceDatabasePopulator;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * PostgreSQL 持久化集成测试（真实 PG 容器）：schema.sql 幂等、画像 upsert、JSONB 多维查询、题库读写。
 * 本机无 Docker 时自动跳过。
 */
@Testcontainers(disabledWithoutDocker = true)
class PostgresPersistenceTest {

    @Container
    static final PostgreSQLContainer<?> PG = new PostgreSQLContainer<>("postgres:16-alpine");

    static NamedParameterJdbcTemplate jdbc;
    PostgresStore store;
    InterviewRecordRepository repository;
    QuestionBankStore questionBank;

    @BeforeAll
    static void initSchema() {
        DriverManagerDataSource ds = new DriverManagerDataSource(PG.getJdbcUrl(), PG.getUsername(), PG.getPassword());
        ResourceDatabasePopulator populator = new ResourceDatabasePopulator(new ClassPathResource("db/schema.sql"));
        populator.execute(ds);
        populator.execute(ds); // 幂等：重复执行不报错
        jdbc = new NamedParameterJdbcTemplate(ds);
    }

    @BeforeEach
    void setUp() {
        jdbc.getJdbcTemplate().execute("TRUNCATE user_profiles, interview_records, question_bank");
        store = new PostgresStore(jdbc);
        repository = new InterviewRecordRepository(jdbc);
        questionBank = new QuestionBankStore(jdbc);
    }

    @Test
    @DisplayName("用户画像 upsert 与读取（JSONB 往返）")
    void profileRoundTrip() {
        UserProfile p = UserProfile.builder().userId("u1").name("alice")
                .skillLevel(Map.of("redis", "advanced"))
                .weakPoints(List.of(UserProfile.WeakPoint.builder().topic("MVCC").score(40).hitCount(1)
                        .wrongCount(1).lastSeen(LocalDateTime.now()).build()))
                .build();
        store.saveProfile(p);
        p.setName("alice2");
        store.saveProfile(p);

        UserProfile loaded = store.loadProfile("u1");
        assertEquals("alice2", loaded.getName());
        assertEquals("advanced", loaded.getSkillLevel().get("redis"));
        assertEquals("MVCC", loaded.getWeakPoints().get(0).getTopic());
        assertNull(store.loadProfile("missing"));
    }

    @Test
    @DisplayName("面试记录 JSONB 多维查询：评级 / 薄弱点 / 遗漏知识点 / 维度分数 / 分数区间")
    void multiDimensionalSearch() {
        save("s1", "Go 后端", 85, "A", List.of("分布式事务"), Map.of("基础知识", 90.0, "项目经验", 80.0), List.of("2PC"));
        save("s2", "Java 后端", 62, "C", List.of("MySQL 索引", "分布式事务"), Map.of("基础知识", 55.0), List.of("MVCC"));
        save("s3", "Java 后端", 74, "B", List.of("Redis 持久化"), Map.of("基础知识", 70.0), List.of());
        saveFor("other", "s4", 99, "A");

        assertEquals(3, search(InterviewRecordQuery.builder().build()).size());
        assertEquals(List.of("s1"), ids(InterviewRecordQuery.builder().level("a").build()));
        assertEquals(List.of("s2", "s1"), ids(InterviewRecordQuery.builder().weakness("分布式事务").build()));
        assertEquals(List.of("s2"), ids(InterviewRecordQuery.builder().missedKeyPoint("MVCC").build()));
        assertEquals(List.of("s3", "s1"), ids(InterviewRecordQuery.builder().dimension("基础知识").minDimensionScore(70.0).build()));
        assertEquals(List.of("s3"), ids(InterviewRecordQuery.builder().position("java").minScore(70.0).maxScore(80.0).build()));
        assertEquals(List.of("s3"), ids(InterviewRecordQuery.builder().limit(1).build()));

        InterviewRecordSummary detail = repository.findBySession("u1", "s2").orElseThrow();
        assertEquals("C", detail.report().get("overall_level").asText());
        assertEquals(55.0, detail.dimensionScore().get("基础知识"));
        assertTrue(repository.findBySession("u1", "s4").isEmpty(), "不能读到其他用户的记录");

        List<Map<String, Object>> stats = repository.weaknessStats("u1", 10);
        assertEquals("分布式事务", stats.get(0).get("topic"));
        assertEquals(2L, ((Number) stats.get(0).get("count")).longValue());
    }

    @Test
    @DisplayName("GIN 索引可被包含查询使用")
    void ginIndexUsable() {
        for (int i = 0; i < 50; i++) save("x" + i, "p", 60, i % 2 == 0 ? "B" : "C", List.of("t" + i), Map.of(), List.of());
        jdbc.getJdbcTemplate().execute("ANALYZE interview_records");
        // 同一连接内关闭顺序扫描（小表默认会走 seq scan），验证查询能命中 GIN 索引
        String plan = jdbc.getJdbcTemplate().execute((java.sql.Connection conn) -> {
            try (var st = conn.createStatement()) {
                st.execute("SET enable_seqscan = off");
                var rs = st.executeQuery(
                        "EXPLAIN SELECT session_id FROM interview_records WHERE report_json @> '{\"overall_level\":\"B\"}'");
                StringBuilder sb = new StringBuilder();
                while (rs.next()) sb.append(rs.getString(1)).append('\n');
                return sb.toString();
            }
        });
        assertTrue(plan.contains("idx_interview_records_report"), plan);
    }

    @Test
    @DisplayName("题库按来源文件覆盖写入，并转换为 BM25 文档")
    void questionBank() {
        questionBank.replaceSourceFile("u1", "redis.md", List.of(row("q1", "什么是 AOF？"), row("q2", "RDB 原理？")));
        questionBank.replaceSourceFile("u1", "redis.md", List.of(row("q3", "Redis 主从复制？")));
        questionBank.replaceSourceFile("u1", "mysql.md", List.of(row("q4", "B+ 树？")));

        List<RagDocument> docs = questionBank.loadDocuments("u1");
        assertEquals(List.of("q3", "q4"), docs.stream().map(RagDocument::getId).sorted().toList());
        assertTrue(docs.get(0).getContent().contains("\n参考答案："));
        assertEquals(List.of("redis"), docs.stream().filter(d -> d.getId().equals("q3")).findFirst().orElseThrow()
                .getMetadata().get("skills"));
        assertTrue(questionBank.loadDocuments("nobody").isEmpty());
    }

    private QuestionBankStore.QuestionRow row(String id, String q) {
        return QuestionBankStore.QuestionRow.builder().id(id).ownerId("u1").question(q).referenceAnswer("ref")
                .knowledgePoints(List.of("redis")).build();
    }

    private void save(String session, String position, double score, String level, List<String> weaknesses,
                      Map<String, Double> dims, List<String> missed) {
        EvaluationReport report = EvaluationReport.builder().sessionId(session).position(position)
                .overallScore(score).overallLevel(level).weaknesses(weaknesses).dimensionScore(dims)
                .detailedReview(List.of(EvaluationReport.QuestionReview.builder().questionContent("q")
                        .keyPointsMissed(missed).build()))
                .build();
        store.saveInterviewRecord("u1", UserProfile.InterviewRecord.builder().sessionId(session)
                .position(position).overallScore(score).build(), level, report, Map.of("weak_areas", List.of()));
        jdbc.getJdbcTemplate().update("UPDATE interview_records SET created_at = now() + (id || ' seconds')::interval WHERE session_id = ?", session);
    }

    private void saveFor(String user, String session, double score, String level) {
        store.saveInterviewRecord(user, UserProfile.InterviewRecord.builder().sessionId(session).position("p")
                .overallScore(score).build(), level, Map.of("overall_level", level), Map.of());
    }

    private List<InterviewRecordSummary> search(InterviewRecordQuery q) {
        return repository.search("u1", q);
    }

    private List<String> ids(InterviewRecordQuery q) {
        return search(q).stream().map(InterviewRecordSummary::sessionId).toList();
    }
}
