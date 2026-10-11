package com.interview.agent.auth;

import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

/**
 * 从 REST 请求中解析当前登录用户：优先 {@code Authorization: Bearer <jwt>}，其次 {@code ?token=}（EventSource 等无法设置 header 的场景）。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class CurrentUserResolver {

    private final JwtService jwtService;

    public String require(HttpServletRequest request) {
        String token = extractToken(request);
        if (token == null || token.isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "未登录");
        }
        try {
            return jwtService.validateToken(token);
        } catch (Exception e) {
            log.debug("[Auth] token 校验失败: {}", e.getMessage());
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "登录已失效，请重新登录");
        }
    }

    static String extractToken(HttpServletRequest request) {
        String header = request.getHeader("Authorization");
        if (header != null && header.regionMatches(true, 0, "Bearer ", 0, 7)) {
            return header.substring(7).trim();
        }
        return request.getParameter("token");
    }
}
