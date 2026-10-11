package com.interview.agent.agent;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AgentUtilsTest {

    @Test
    void extractsObjectAndArray() {
        assertEquals("{\"a\":1}", AgentUtils.extractJSON("```json\n{\"a\":1}\n```"));
        assertEquals("[{\"a\":1},{\"b\":2}]", AgentUtils.extractJSONArray("结果：[{\"a\":1},{\"b\":2}] 完"));
        // 包裹对象：回退为对象提取，由调用方取数组字段
        assertEquals("{\"questions\":[{\"a\":1}]}", AgentUtils.extractJSONArray("{\"questions\":[{\"a\":1}]}"));
    }
}
