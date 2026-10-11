package com.interview.agent.ml;

import com.interview.agent.config.AppConfig;
import com.interview.agent.model.AnswerScore;
import com.interview.agent.model.Resume;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.*;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

class MlServiceClientTest {

    private record Fixture(MlServiceClient client, MockRestServiceServer server) {}

    private Fixture fixture(boolean parser, boolean scorer) {
        AppConfig.MlProperties props = new AppConfig.MlProperties();
        props.setBaseUrl("http://ml");
        props.setResumeParserEnabled(parser);
        props.setScorerEnabled(scorer);
        RestClient.Builder builder = RestClient.builder().baseUrl("http://ml");
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        return new Fixture(new MlServiceClient(props, builder.build()), server);
    }

    @Test
    @DisplayName("Spring 能选择正确的构造器创建 Bean（多构造器场景）")
    void springCanInstantiate() {
        new ApplicationContextRunner()
                .withBean(AppConfig.class)
                .withBean(MlServiceClient.class)
                .withPropertyValues("app.ml.base-url=http://ml:8000")
                .run(ctx -> {
                    assertNull(ctx.getStartupFailure());
                    assertNotNull(ctx.getBean(MlServiceClient.class));
                });
    }

    @Test
    @DisplayName("未配置 base-url 时所有能力不可用")
    void disabledWithoutBaseUrl() {
        MlServiceClient client = new MlServiceClient(new AppConfig());
        assertFalse(client.resumeParserAvailable());
        assertFalse(client.scorerAvailable());
        assertTrue(client.parseResume("text").isEmpty());
        assertTrue(client.score("q", "a", "r").isEmpty());
    }

    @Test
    @DisplayName("简历解析：反序列化结构化字段，忽略未知字段")
    void parsesResume() {
        Fixture f = fixture(true, false);
        f.server().expect(requestTo("http://ml/v1/resume/parse"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(jsonPath("$.text").value("张三 Go 工程师"))
                .andRespond(withSuccess("""
                        {"name":"张三","skills":["Go","Redis"],"chunks":3,
                         "projects":[{"name":"IM","tech_stack":["Go"],"highlights":["百万连接"]}]}
                        """, MediaType.APPLICATION_JSON));

        Resume r = f.client().parseResume("张三 Go 工程师").orElseThrow();
        assertEquals("张三", r.getName());
        assertEquals(2, r.getSkills().size());
        assertEquals("Go", r.getProjects().get(0).getTechStack().get(0));
        f.server().verify();
    }

    @Test
    @DisplayName("评分模型：成功返回分数，服务异常时降级为空")
    void scoresAndFallsBack() {
        Fixture f = fixture(false, true);
        f.server().expect(requestTo("http://ml/v1/score"))
                .andRespond(withSuccess("{\"score\":72,\"feedback\":\"ok\",\"key_points_hit\":[\"a\"],\"should_follow_up\":true}",
                        MediaType.APPLICATION_JSON));
        f.server().expect(requestTo("http://ml/v1/score")).andRespond(withServerError());

        AnswerScore s = f.client().score("q", "a", "r").orElseThrow();
        assertEquals(72, s.getScore());
        assertTrue(s.isShouldFollowUp());
        assertEquals(Optional.empty(), f.client().score("q", "a", "r"));
        assertTrue(f.client().parseResume("x").isEmpty(), "解析未启用时不发请求");
        f.server().verify();
    }
}
