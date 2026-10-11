package com.interview.agent.agent;

/**
 * Agent 通用工具方法
 */
public class AgentUtils {

    /**
     * 从 LLM 响应文本中提取 JSON 内容。
     * 处理 markdown 代码块包裹的情况，提取第一个 { 到最后一个 } 之间的内容。
     */
    public static String extractJSON(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }

        // 找到第一个 {
        int start = -1;
        for (int i = 0; i < text.length(); i++) {
            if (text.charAt(i) == '{') {
                start = i;
                break;
            }
        }
        if (start == -1) {
            return text;
        }

        // 找到最后一个 }
        int end = -1;
        for (int i = text.length() - 1; i >= start; i--) {
            if (text.charAt(i) == '}') {
                end = i + 1;
                break;
            }
        }
        if (end == -1) {
            return text;
        }

        return text.substring(start, end);
    }

    /**
     * 从 LLM 响应文本中提取 JSON 数组（第一个 [ 到最后一个 ]）；
     * 若模型返回 {"questions": [...]} 之类的包裹对象，则回退到对象提取，由调用方处理。
     */
    public static String extractJSONArray(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }
        int start = text.indexOf('[');
        int end = text.lastIndexOf(']');
        int objStart = text.indexOf('{');
        if (start == -1 || end <= start || (objStart != -1 && objStart < start)) {
            return extractJSON(text);
        }
        return text.substring(start, end + 1);
    }
}
