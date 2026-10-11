package com.interview.agent.chat;

import com.interview.agent.agent.ChatAgent;
import com.interview.agent.auth.CurrentUserResolver;
import com.interview.agent.auth.JwtService;
import com.interview.agent.skill.Skill;
import com.interview.agent.skill.SkillRegistry;
import com.interview.agent.skill.SkillResponse;
import com.interview.agent.skill.SkillState;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.Message;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import reactor.core.publisher.Flux;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class ChatStreamControllerTest {

    private ChatAgent chatAgent;
    private JwtService jwtService;
    private SkillRegistry registry;
    private MockMvc mvc;
    private final List<List<Message>> seenHistories = new ArrayList<>();

    @BeforeEach
    void setUp() {
        chatAgent = mock(ChatAgent.class);
        jwtService = mock(JwtService.class);
        when(jwtService.validateToken("good")).thenReturn("alice");
        when(jwtService.validateToken("bad")).thenThrow(new RuntimeException("invalid"));
        when(chatAgent.stream(anyList(), anyString())).thenAnswer(inv -> {
            seenHistories.add(new ArrayList<>(inv.getArgument(0)));
            return Flux.just("你好", "，", "世界");
        });

        registry = new SkillRegistry();
        registry.register(new Skill() {
            @Override public String name() { return "echo"; }
            @Override public String description() { return "echo"; }
            @Override public boolean match(String input) { return input.startsWith("/echo"); }
            @Override public SkillResponse handle(String input, SkillState state) {
                return SkillResponse.builder().content("ECHO:" + input).done(true).state(state).build();
            }
        });

        ChatDispatcher dispatcher = new ChatDispatcher(registry, chatAgent);
        mvc = MockMvcBuilders.standaloneSetup(
                new ChatStreamController(dispatcher, chatAgent, new CurrentUserResolver(jwtService))).build();
    }

    private String stream(String token, String body) throws Exception {
        MvcResult started = mvc.perform(post("/api/chat/stream")
                        .header("Authorization", "Bearer " + token)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(request().asyncStarted())
                .andReturn();
        return mvc.perform(asyncDispatch(started)).andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString(StandardCharsets.UTF_8);
    }

    @Test
    @DisplayName("LLM 增量输出逐段推送 delta，最后推送 done，并写入短期记忆")
    void streamsDeltasAndRemembersHistory() throws Exception {
        String out = stream("good", "{\"content\":\"hi\",\"conversation_id\":\"c1\"}");
        assertTrue(out.contains("event:delta"), out);
        assertTrue(out.contains("\"content\":\"你好\""), out);
        assertTrue(out.contains("event:done"), out);
        assertTrue(out.contains("\"content\":\"你好，世界\""), out);
        assertTrue(out.contains("\"source\":\"llm\""), out);

        stream("good", "{\"content\":\"again\",\"conversation_id\":\"c1\"}");
        assertEquals(0, seenHistories.get(0).size());
        assertEquals(2, seenHistories.get(1).size(), "第二轮应带上第一轮的 user + assistant 消息");

        stream("good", "{\"content\":\"new\",\"conversation_id\":\"c2\"}");
        assertEquals(0, seenHistories.get(2).size(), "不同会话互相隔离");
    }

    @Test
    @DisplayName("命中 Skill 时一次性返回 Skill 回复，不调用 LLM")
    void skillShortCircuits() throws Exception {
        String out = stream("good", "{\"content\":\"/echo x\"}");
        assertTrue(out.contains("ECHO:/echo x"), out);
        assertTrue(out.contains("\"source\":\"skill\""), out);
        verify(chatAgent, never()).stream(anyList(), anyString());
    }

    @Test
    @DisplayName("未登录 / token 无效返回 401，空内容返回 400")
    void authAndValidation() throws Exception {
        mvc.perform(post("/api/chat/stream").contentType(MediaType.APPLICATION_JSON).content("{\"content\":\"hi\"}"))
                .andExpect(status().isUnauthorized());
        mvc.perform(post("/api/chat/stream").header("Authorization", "Bearer bad")
                        .contentType(MediaType.APPLICATION_JSON).content("{\"content\":\"hi\"}"))
                .andExpect(status().isUnauthorized());
        mvc.perform(post("/api/chat/stream").header("Authorization", "Bearer good")
                        .contentType(MediaType.APPLICATION_JSON).content("{\"content\":\"  \"}"))
                .andExpect(status().isBadRequest());
    }
}
