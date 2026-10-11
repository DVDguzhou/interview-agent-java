package com.interview.agent.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Data
@Configuration
@ConfigurationProperties(prefix = "app")
public class AppConfig {

    private MilvusProperties milvus = new MilvusProperties();
    private JwtProperties jwt = new JwtProperties();
    private GitHubProperties github = new GitHubProperties();
    private AuthProperties auth = new AuthProperties();
    private MlProperties ml = new MlProperties();

    /** Python ML 服务（简历解析 / 微调评分模型），baseUrl 为空表示未部署 */
    @Data
    public static class MlProperties {
        private String baseUrl = "";
        private int timeoutSeconds = 60;
        private boolean resumeParserEnabled = true;
        private boolean scorerEnabled = false;
    }

    @Data
    public static class MilvusProperties {
        private String host = "localhost";
        private int port = 19530;
    }

    @Data
    public static class JwtProperties {
        private String secret = "interview-agent-default-secret";
        private long expiration = 86400000; // 24 hours
    }

    @Data
    public static class GitHubProperties {
        private String token = "";
    }

    @Data
    public static class AuthProperties {
        private boolean enabled = true;
    }
}
