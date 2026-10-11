package com.interview.agent.rag;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class BM25ManagerTest {

    private static RagDocument doc(String id, String content) {
        return RagDocument.builder().id(id).content(content).build();
    }

    @Test
    @DisplayName("首次检索从题库懒加载；题库变化后 reload 立即生效")
    void lazyLoadAndReload() {
        QuestionBankStore store = mock(QuestionBankStore.class);
        List<RagDocument> bank = new ArrayList<>(List.of(doc("q1", "redis 持久化 AOF RDB")));
        when(store.loadDocuments("u1")).thenAnswer(inv -> List.copyOf(bank));

        BM25Manager m = new BM25Manager();
        m.setQuestionBankStore(store);
        assertEquals("q1", m.retrieve("u1", "redis aof").get(0).getId());
        m.retrieve("u1", "redis");
        verify(store, times(1)).loadDocuments("u1");

        bank.add(doc("q2", "mysql mvcc 隔离级别"));
        m.reload("u1");
        assertEquals("q2", m.retrieve("u1", "mysql mvcc").get(0).getId());
    }

    @Test
    @DisplayName("replaceDocuments 写入的索引不会被题库刷新覆盖")
    void pinnedIndexNotOverwritten() {
        QuestionBankStore store = mock(QuestionBankStore.class);
        when(store.loadDocuments(anyString())).thenReturn(List.of());
        BM25Manager m = new BM25Manager();
        m.setQuestionBankStore(store);

        m.replaceDocuments("eval", List.of(doc("e1", "kafka 消息积压")));
        assertEquals("e1", m.retrieve("eval", "kafka").get(0).getId());
        verify(store, never()).loadDocuments("eval");
    }

    @Test
    @DisplayName("未配置题库存储时退化为纯内存索引")
    void worksWithoutStore() {
        BM25Manager m = new BM25Manager();
        assertTrue(m.retrieve("u", "x").isEmpty());
        m.appendDocuments("u", List.of(doc("a", "go channel 原理")));
        assertEquals("a", m.retrieve("u", "channel").get(0).getId());
    }
}
