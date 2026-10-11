package com.interview.agent.record;

import lombok.Builder;

/**
 * 面试记录多维查询条件，所有字段可选（null 表示不过滤）。
 *
 * @param position          岗位名模糊匹配（ILIKE）
 * @param minScore          综合分下限
 * @param maxScore          综合分上限
 * @param level             评级（A/B/C/D），走 report_json GIN 包含查询
 * @param weakness          报告 weaknesses 数组中的元素（精确匹配，GIN）
 * @param missedKeyPoint    逐题点评中遗漏的知识点（精确匹配，GIN）
 * @param dimension         维度名（dimension_score 的 key）
 * @param minDimensionScore 该维度分数下限（需与 dimension 同时提供）
 * @param limit             返回条数，默认 20，上限 100
 * @param offset            分页偏移
 */
@Builder
public record InterviewRecordQuery(
        String position,
        Double minScore,
        Double maxScore,
        String level,
        String weakness,
        String missedKeyPoint,
        String dimension,
        Double minDimensionScore,
        Integer limit,
        Integer offset) {

    public int effectiveLimit() {
        return limit == null || limit <= 0 ? 20 : Math.min(limit, 100);
    }

    public int effectiveOffset() {
        return offset == null || offset < 0 ? 0 : offset;
    }
}
