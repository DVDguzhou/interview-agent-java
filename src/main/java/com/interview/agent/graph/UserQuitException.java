package com.interview.agent.graph;

/**
 * 用户主动终止面试异常
 */
public class UserQuitException extends Exception {
    public UserQuitException() {
        super("用户主动终止面试");
    }
}
