package com.interview.agent.chat;

import com.interview.agent.agent.ChatAgent;
import com.interview.agent.skill.*;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * 日常对话分发（WebSocket 与 SSE 共用）：
 * <ol>
 *   <li>已激活的 Skill 继续处理（有状态多轮交互）；</li>
 *   <li>SkillRegistry 按优先级匹配新 Skill；</li>
 *   <li>ChatAgent 兜底，携带 20 条滑动窗口的短期对话记忆。</li>
 * </ol>
 */
@Component
@RequiredArgsConstructor
public class ChatDispatcher {

    private final SkillRegistry skillRegistry;
    private final ChatAgent chatAgent;

    /** 单个会话的对话状态 */
    public static final class Conversation {
        final String userId;
        final List<Message> history = new ArrayList<>();
        Skill activeSkill;
        SkillState skillState;

        public Conversation(String userId) {
            this.userId = userId;
        }

        public synchronized List<Message> historySnapshot() {
            return new ArrayList<>(history);
        }
    }

    /**
     * 尝试由 Skill 处理本轮输入。
     *
     * @return Skill 的回复；返回 null 表示应交给 ChatAgent
     */
    public synchronized String trySkill(Conversation c, String input) {
        if (c.activeSkill != null) {
            if (SkillUtils.isQuitCommand(input) || (c.skillState != null && c.skillState.isExpired())) {
                c.activeSkill = null;
                c.skillState = null;
                return "已退出技能模式。";
            }
            // 技能会话进行中，若用户又发出明确的新技能/新测验意图（例如测验做到一半再说「来几道 mysql 面试题」），
            // 则结束当前会话、切换到新技能，而不是把这句话当成当前题目的回答。普通的答题内容不含触发词，不会被误切。
            if (skillRegistry.match(input) == null) {
                return runSkill(c, c.activeSkill, input);
            }
            c.activeSkill = null;
            c.skillState = null;
        }

        Skill matched = skillRegistry.match(input);
        if (matched == null) {
            return null;
        }
        c.activeSkill = matched;
        c.skillState = SkillState.create(matched.name());
        c.skillState.setUserId(c.userId);
        return runSkill(c, matched, input);
    }

    /** 同步对话：Skill 优先，否则 ChatAgent */
    public String chat(Conversation c, String input) {
        String reply = trySkill(c, input);
        if (reply != null) {
            return reply;
        }
        reply = chatAgent.chat(c.historySnapshot(), input);
        remember(c, input, reply);
        return reply;
    }

    /** 记录一轮问答到短期记忆（滑动窗口） */
    public void remember(Conversation c, String input, String reply) {
        synchronized (c) {
            c.history.add(new UserMessage(input));
            c.history.add(new AssistantMessage(reply));
            int overflow = c.history.size() - ChatAgent.MAX_HISTORY_SIZE;
            if (overflow > 0) {
                c.history.subList(0, overflow).clear();
            }
        }
    }

    private String runSkill(Conversation c, Skill skill, String input) {
        SkillResponse resp = skill.handle(input, c.skillState);
        c.skillState = resp.getState();
        if (resp.isDone()) {
            c.activeSkill = null;
            c.skillState = null;
        }
        return resp.getContent();
    }
}
