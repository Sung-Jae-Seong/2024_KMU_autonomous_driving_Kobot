class PID:

    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

        self.cte_prev = 0.0
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0
        # 적분오차 제한값 설정
        self.i_min = -10
        self.i_max = 10

    def reset(self):
        self.cte_prev = 0.0
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def pid_control(self, cte):
        self.d_error = cte - self.cte_prev
        self.p_error = cte
        self.i_error += cte
        self.i_error = max(min(self.i_error, self.i_max), self.i_min)

        self.cte_prev = cte
        # PID 제어 출력 계산
        return self.Kp * self.p_error + self.Ki * self.i_error + self.Kd * self.d_error
