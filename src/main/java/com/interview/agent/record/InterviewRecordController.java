package com.interview.agent.record;

import com.interview.agent.auth.CurrentUserResolver;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.Map;

/**
 * 面试历史 / 评估报告查询接口（前端「历史报告」页使用）。
 * <pre>
 * GET /api/interviews                      列表 + 多维筛选（见 {@link InterviewRecordQuery}）
 * GET /api/interviews/{sessionId}          单场详情（完整报告与复习计划）
 * GET /api/interviews/stats/weaknesses     薄弱点频次统计
 * </pre>
 */
@RestController
@RequestMapping("/api/interviews")
@RequiredArgsConstructor
public class InterviewRecordController {

    private final InterviewRecordRepository repository;
    private final CurrentUserResolver currentUser;

    @GetMapping
    public List<InterviewRecordSummary> list(HttpServletRequest request,
                                             @RequestParam(required = false) String position,
                                             @RequestParam(name = "min_score", required = false) Double minScore,
                                             @RequestParam(name = "max_score", required = false) Double maxScore,
                                             @RequestParam(required = false) String level,
                                             @RequestParam(required = false) String weakness,
                                             @RequestParam(name = "missed_key_point", required = false) String missedKeyPoint,
                                             @RequestParam(required = false) String dimension,
                                             @RequestParam(name = "min_dimension_score", required = false) Double minDimensionScore,
                                             @RequestParam(required = false) Integer limit,
                                             @RequestParam(required = false) Integer offset) {
        String userId = currentUser.require(request);
        return repository.search(userId, InterviewRecordQuery.builder()
                .position(position).minScore(minScore).maxScore(maxScore).level(level)
                .weakness(weakness).missedKeyPoint(missedKeyPoint)
                .dimension(dimension).minDimensionScore(minDimensionScore)
                .limit(limit).offset(offset)
                .build());
    }

    @GetMapping("/stats/weaknesses")
    public List<Map<String, Object>> weaknessStats(HttpServletRequest request,
                                                   @RequestParam(defaultValue = "10") int limit) {
        return repository.weaknessStats(currentUser.require(request), limit);
    }

    @GetMapping("/{sessionId}")
    public InterviewRecordSummary detail(HttpServletRequest request, @PathVariable String sessionId) {
        return repository.findBySession(currentUser.require(request), sessionId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "面试记录不存在"));
    }
}
