package com.interview.agent.ml;

import com.interview.agent.config.AppConfig;
import com.interview.agent.model.AnswerScore;
import com.interview.agent.model.Resume;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.time.Duration;
import java.util.Map;
import java.util.Optional;

/**
 * Python ML 服务客户端（见仓库 ml/ 目录，FastAPI）：
 * <ul>
 *   <li>{@code POST /v1/resume/parse}：LlamaIndex 文档解析 + 分块 + Prompt 抽取结构化简历；</li>
 *   <li>{@code POST /v1/score}：QLoRA 微调的面试回答评分模型。</li>
 * </ul>
 * 所有调用失败均返回 {@link Optional#empty()}，由调用方降级到通用大模型链路，不影响面试主流程。
 */
@Slf4j
@Component
public class MlServiceClient {

    private final AppConfig.MlProperties props;
    private final RestClient restClient;

    @Autowired
    public MlServiceClient(AppConfig appConfig) {
        this.props = appConfig.getMl();
        if (props.getBaseUrl() == null || props.getBaseUrl().isBlank()) {
            this.restClient = null;
            return;
        }
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofSeconds(5));
        factory.setReadTimeout(Duration.ofSeconds(props.getTimeoutSeconds()));
        this.restClient = RestClient.builder()
                .baseUrl(props.getBaseUrl())
                .requestFactory(factory)
                .build();
        log.info("[ML] 已配置 ML 服务: {}（简历解析={}, 微调评分={}）",
                props.getBaseUrl(), props.isResumeParserEnabled(), props.isScorerEnabled());
    }

    MlServiceClient(AppConfig.MlProperties props, RestClient restClient) {
        this.props = props;
        this.restClient = restClient;
    }

    public boolean resumeParserAvailable() {
        return restClient != null && props.isResumeParserEnabled();
    }

    public boolean scorerAvailable() {
        return restClient != null && props.isScorerEnabled();
    }

    /** 结构化解析简历文本 */
    public Optional<Resume> parseResume(String resumeText) {
        if (!resumeParserAvailable() || resumeText == null || resumeText.isBlank()) {
            return Optional.empty();
        }
        return call("resume/parse", () -> restClient.post().uri("/v1/resume/parse")
                .contentType(MediaType.APPLICATION_JSON)
                .body(Map.of("text", resumeText))
                .retrieve()
                .body(Resume.class));
    }

    /** 调用微调评分模型为单题回答打分 */
    public Optional<AnswerScore> score(String question, String answer, String reference) {
        if (!scorerAvailable()) {
            return Optional.empty();
        }
        return call("score", () -> restClient.post().uri("/v1/score")
                .contentType(MediaType.APPLICATION_JSON)
                .body(Map.of(
                        "question", question != null ? question : "",
                        "answer", answer != null ? answer : "",
                        "reference", reference != null ? reference : ""))
                .retrieve()
                .body(AnswerScore.class));
    }

    private <T> Optional<T> call(String name, java.util.function.Supplier<T> fn) {
        try {
            return Optional.ofNullable(fn.get());
        } catch (Exception e) {
            log.warn("[ML] {} 调用失败，降级到 LLM 链路: {}", name, e.getMessage());
            return Optional.empty();
        }
    }
}
