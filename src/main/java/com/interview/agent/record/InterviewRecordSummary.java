package com.interview.agent.record;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;

/**
 * 面试记录视图。列表接口只返回摘要字段，详情接口额外带上完整 report / review_plan。
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record InterviewRecordSummary(
        @JsonProperty("session_id") String sessionId,
        String position,
        @JsonProperty("overall_score") double overallScore,
        @JsonProperty("overall_level") String overallLevel,
        @JsonProperty("dimension_score") Map<String, Double> dimensionScore,
        List<String> weaknesses,
        @JsonProperty("created_at") OffsetDateTime createdAt,
        JsonNode report,
        @JsonProperty("review_plan") JsonNode reviewPlan) {
}
