/**
 * @author: 公众号：IT杨秀才
 * @doc: AI模拟面试官 - Java版（Spring AI Alibaba）
 */
package com.interview.agent.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ResumeMatchResult {

    @JsonProperty("overall_score")
    private double overallScore;

    @JsonProperty("skill_match")
    private List<SkillMatch> skillMatch;

    private List<String> strengths;
    private List<String> weaknesses;

    @JsonProperty("focus_areas")
    private List<String> focusAreas;

    @JsonProperty("resume_gaps")
    private List<String> resumeGaps;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SkillMatch {
        @JsonProperty("skill_name")
        private String skillName;

        private boolean required;
        private boolean matched;

        @JsonProperty("match_score")
        private double matchScore;

        private String evidence;
    }
}
