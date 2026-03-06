import numpy as np
import math

class UGV:
    def __init__(self):
        #kinematic_state = self.get_state()

        self.x = 0
        self.y = 0
        self.yaw = 0

        self.v = 0
        self.w = 0


        self.bound_x = [0, 50]
        self.bound_y = [0, 50]

        self.target_x = [0, 50]
        self.target_y = [0, 50]

        self.d_safe = 5

        self.tx, self.ty= self.generate_target()
        self.init_distance = self.get_distance()

        self.obstacle = [20,30]

        self.done = False

    def generate_target(self):
        """
        生成目标点的位置
        seed为随机种子
        """
        tx = np.random.rand() * (self.target_x[1] - self.target_x[0]) + self.target_x[0]
        ty = np.random.rand() * (self.target_y[1] - self.target_y[0]) + self.target_y[0]

        return tx, ty

    def get_distance(self):
        return math.sqrt((self.x - self.tx) ** 2 + (self.y - self.ty) ** 2)

    def get_deflection_angle(self):
        # 连线向量
        ax = self.tx - self.x
        ay = self.ty - self.y

        if ax == 0 and ay == 0:
            return 0.0

        # 速度方向向量
        bx = self.v * math.cos(self.yaw)
        by = self.v * math.sin(self.yaw)

        if self.v == 0:
            return 180

        model_a = math.hypot(ax, ay)  # 优化：使用内置函数计算欧氏距离
        model_b = abs(self.v)

        cos_ab = (ax * bx + ay * by) / (model_a * model_b)
        cos_ab = max(-1.0, min(1.0, cos_ab))

        # 计算角度
        radius = math.acos(cos_ab)
        angle = np.rad2deg(radius)

        return angle


    def get_state(self):
        # 进行归一化
        position = np.array([self.tx - self.x, self.ty - self.y])
        #target = np.array([self.get_distance() / self.init_distance])
        velocity = np.array([self.v,self.w])
        angle = np.array([self.get_deflection_angle() / 180])
        #sensor_data = np.array(self.get_distance_sensors_data()) / 20

        state = np.append(position, velocity)
        state = np.append(state, velocity)
        state = np.append(state, angle)

        return state

    def step(self, action):
        # 保存上一时刻的距离，用于计算进度奖励
        prev_distance = self.get_distance()

        self.v = action[0]
        self.w = action[1]

        # 运动学更新 (dt = 0.1)
        dt = 0.1
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt

        # 计算当前距离
        curr_distance = self.get_distance()

        # 初始化奖励和结束标志
        reward = 0
        done = False
        # 1. 到达目标奖励
        if curr_distance <= 5.0:
            reward += 100
            done = True

        # 2. 碰撞惩罚
        elif self.has_collided():
            reward -= 100
            done = True

        # 3. 越界惩罚
        elif self.x < self.bound_x[0] or self.x > self.bound_x[1] or \
             self.y < self.bound_y[0] or self.y > self.bound_y[1]:
            reward -= 50
            done = True

        else:
            # 4. 进度奖励 (靠近目标��正向奖励)
            reward += 10 * (prev_distance - curr_distance)

            # 5. 时间步惩罚 (鼓励快速到达)
            reward -= 0.1

            # 6. 角度惩罚 (鼓励朝向目标)
            angle_diff = self.get_deflection_angle()
            reward -= 0.1 * abs(angle_diff)

        self.done = done
        next_state = self.get_state()
        reward -= (self.v ** 2 + self.w ** 2)  # 控制输入惩罚，鼓励节能

        return next_state, reward, done

    def has_collided(self):
        if math.sqrt((self.x - self.obstacle[0]) ** 2 + (self.y - self.obstacle[1]) ** 2) <= self.d_safe:
            return True
        return False

if __name__ == '__main__':
    env = UGV()
    state = env.get_state()
    print(state)
    action = np.array([1.0, 0.5])
    next_state, reward, done = env.step(action)
    print(next_state, reward, done)