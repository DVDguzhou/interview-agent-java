package com.interview.agent.chat;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.interview.agent.agent.ChatAgent;
import com.interview.agent.auth.CurrentUserResolver;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import reactor.core.Disposable;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 日常对话流式接口（SSE）。
 * <pre>
 * POST /api/chat/stream   {"content": "...", "conversation_id": "..."}
 *   event: delta  data: {"content": "增量文本"}
 *   event: done   data: {"content": "完整回复", "source": "llm|skill"}
 *   event: error  data: {"message": "..."}
 * DELETE /api/chat/conversations/{id}   清空会话短期记忆
 * </pre>
 * Skill 命中时一次性返回（Skill 为有状态多轮交互，输出依赖内部状态机）；其余走 ChatAgent 流式输出。
 */
@Slf4j
@RestController
@RequestMapping("/api/chat")
public class ChatStreamController {

    private static final long EMITTER_TIMEOUT_MS = 5 * 60 * 1000L;
    private static final int MAX_CONVERSATIONS = 2000;

    private final ChatDispatcher dispatcher;
    private final ChatAgent chatAgent;
    private final CurrentUserResolver currentUser;

    /** 会话 LRU：key = userId + ":" + conversationId */
    private final Map<String, ChatDispatcher.Conversation> conversations =
            Collections.synchronizedMap(new LinkedHashMap<>(16, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<String, ChatDispatcher.Conversation> eldest) {
                    return size() > MAX_CONVERSATIONS;
                }
            });

    public ChatStreamController(ChatDispatcher dispatcher, ChatAgent chatAgent, CurrentUserResolver currentUser) {
        this.dispatcher = dispatcher;
        this.chatAgent = chatAgent;
        this.currentUser = currentUser;
    }

    public record ChatStreamRequest(String content, @JsonProperty("conversation_id") String conversationId) {
    }

    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(HttpServletRequest request, @RequestBody ChatStreamRequest body) {
        String userId = currentUser.require(request);
        if (body == null || body.content() == null || body.content().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "content 不能为空");
        }
        String input = body.content();
        ChatDispatcher.Conversation conv = conversations.computeIfAbsent(
                key(userId, body.conversationId()), k -> new ChatDispatcher.Conversation(userId));

        SseEmitter emitter = new SseEmitter(EMITTER_TIMEOUT_MS);

        String skillReply;
        try {
            skillReply = dispatcher.trySkill(conv, input);
        } catch (Exception e) {
            log.error("[SSE] Skill 处理异常: {}", e.getMessage(), e);
            send(emitter, "error", Map.of("message", "处理失败: " + e.getMessage()));
            emitter.complete();
            return emitter;
        }
        if (skillReply != null) {
            send(emitter, "delta", Map.of("content", skillReply));
            send(emitter, "done", Map.of("content", skillReply, "source", "skill"));
            emitter.complete();
            return emitter;
        }

        StringBuilder full = new StringBuilder();
        Disposable subscription = chatAgent.stream(conv.historySnapshot(), input).subscribe(
                delta -> {
                    full.append(delta);
                    send(emitter, "delta", Map.of("content", delta));
                },
                err -> {
                    log.error("[SSE] 流式生成失败: {}", err.getMessage());
                    send(emitter, "error", Map.of("message", "生成失败: " + err.getMessage()));
                    emitter.complete();
                },
                () -> {
                    dispatcher.remember(conv, input, full.toString());
                    send(emitter, "done", Map.of("content", full.toString(), "source", "llm"));
                    emitter.complete();
                });
        emitter.onTimeout(subscription::dispose);
        emitter.onError(e -> subscription.dispose());
        return emitter;
    }

    @DeleteMapping("/conversations/{conversationId}")
    public Map<String, Object> clear(HttpServletRequest request, @PathVariable String conversationId) {
        conversations.remove(key(currentUser.require(request), conversationId));
        return Map.of("ok", true);
    }

    private static String key(String userId, String conversationId) {
        return userId + ":" + (conversationId == null || conversationId.isBlank() ? "default" : conversationId);
    }

    private static void send(SseEmitter emitter, String event, Object data) {
        try {
            emitter.send(SseEmitter.event().name(event).data(data, MediaType.APPLICATION_JSON));
        } catch (IOException | IllegalStateException e) {
            log.debug("[SSE] 客户端已断开: {}", e.getMessage());
        }
    }
}
