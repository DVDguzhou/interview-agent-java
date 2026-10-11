package com.interview.agent.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * BM25 管理器：按用户管理 BM25 索引实例。
 * <p>
 * 索引常驻内存；若注入了 {@link QuestionBankStore}，首次检索某用户时会从 PostgreSQL 题库懒加载重建，
 * 并在超过 {@link #STORE_REFRESH_INTERVAL} 后自动刷新，使离线 ETL 管道直接写库的题目无需重启即可被检索到；
 * 在线上传题库后调用 {@link #reload(String)} 立即全量重建，避免同名文件重传后残留旧题。
 */
@Slf4j
@Component
public class BM25Manager {

    /** 题库存储驱动的索引刷新间隔 */
    static final Duration STORE_REFRESH_INTERVAL = Duration.ofMinutes(5);

    private final Map<String, BM25Retriever> retrievers = new ConcurrentHashMap<>();
    private final Map<String, Instant> loadedAt = new ConcurrentHashMap<>();
    /** 通过 replaceDocuments 直接写入的索引（如离线评估），不受题库存储刷新影响 */
    private final Set<String> pinned = ConcurrentHashMap.newKeySet();
    private final int topK;
    private QuestionBankStore questionBankStore;

    public BM25Manager() {
        this.topK = 10;
    }

    public BM25Manager(int topK) {
        this.topK = topK > 0 ? topK : 10;
    }

    @Autowired(required = false)
    public void setQuestionBankStore(QuestionBankStore questionBankStore) {
        this.questionBankStore = questionBankStore;
    }

    /**
     * 覆盖指定用户的题库索引
     */
    public void replaceDocuments(String userID, List<RagDocument> docs) {
        BM25Retriever r = new BM25Retriever(topK);
        r.indexDocuments(docs);
        retrievers.put(userID, r);
        pinned.add(userID);
    }

    /**
     * 追加文档到指定用户的索引
     */
    public void appendDocuments(String userID, List<RagDocument> docs) {
        retrieverFor(userID).appendDocuments(docs);
    }

    /**
     * 从持久化题库全量重建指定用户的索引（未配置题库存储时为 no-op）
     */
    public void reload(String userID) {
        if (questionBankStore == null) return;
        pinned.remove(userID);
        retrievers.put(userID, loadFromStore(userID));
    }

    /**
     * 检索指定用户的题库
     */
    public List<RagDocument> retrieve(String userID, String query) {
        return retrieverFor(userID).retrieve(query);
    }

    /**
     * 删除指定用户的索引
     */
    public void deleteByUserID(String userID) {
        retrievers.remove(userID);
        loadedAt.remove(userID);
        pinned.remove(userID);
    }

    private BM25Retriever retrieverFor(String userID) {
        if (questionBankStore != null && !pinned.contains(userID)) {
            Instant t = loadedAt.get(userID);
            if (t == null || Duration.between(t, Instant.now()).compareTo(STORE_REFRESH_INTERVAL) > 0) {
                reload(userID);
            }
        }
        return retrievers.computeIfAbsent(userID, k -> new BM25Retriever(topK));
    }

    private BM25Retriever loadFromStore(String userID) {
        loadedAt.put(userID, Instant.now());
        BM25Retriever r = new BM25Retriever(topK);
        if (questionBankStore == null) return r;
        try {
            List<RagDocument> docs = questionBankStore.loadDocuments(userID);
            if (!docs.isEmpty()) {
                r.indexDocuments(docs);
                log.info("[BM25] 从题库加载用户 {} 的 {} 道题目", userID, docs.size());
            }
        } catch (Exception e) {
            log.warn("[BM25] 加载用户 {} 题库失败（使用空索引）: {}", userID, e.getMessage());
        }
        return r;
    }
}
