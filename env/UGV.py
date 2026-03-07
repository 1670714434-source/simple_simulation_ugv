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


        self.bound_x = [0, 80]
        self.bound_y = [0, 80]

        self.target_x = [0, 80]
        self.target_y = [0, 80]

        self.d_safe = 5



        self.tx, self.ty= self.generate_target()

        ##################定义领导者####################
        self.leader_x, self.leader_y = self.generate_target()
        self.leader_yaw = 0  # 领导者初始航向角
        self.leader_v = 3    # 领导者速度

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

    def update_leader(self, dt):
        """
        更新领导者位置：随机游走
        """
        # 领导者策略：随机游走 (随机改变航向)
        # 随机转向范围：[-1.0, 1.0] rad/s * dt
        self.leader_yaw += np.random.uniform(-1.0, 1.0) * dt
        self.leader_v += np.random.uniform(-1.0, 1.0) * dt
        # 计算新位置
        self.leader_x += self.leader_v * math.cos(self.leader_yaw) * dt
        self.leader_y += self.leader_v * math.sin(self.leader_yaw) * dt

        # 简单的边界限制，防止跑出地图
        # 如果超出边界，不仅要截断，最好让它掉头，避免卡在边缘
        if self.leader_x < self.bound_x[0] or self.leader_x > self.bound_x[1]:
            self.leader_yaw = math.pi - self.leader_yaw  # 水平反射
            self.leader_x = np.clip(self.leader_x, self.bound_x[0], self.bound_x[1])

        if self.leader_y < self.bound_y[0] or self.leader_y > self.bound_y[1]:
            self.leader_yaw = -self.leader_yaw  # 垂直反射
            self.leader_y = np.clip(self.leader_y, self.bound_y[0], self.bound_y[1])

        # 将目标点(tx, ty) 实时更新为领导者位置
        self.tx = self.leader_x
        self.ty = self.leader_y

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
        #velocity = np.array([self.v,self.w])
        angle = np.array([self.get_deflection_angle() / 180])
        #sensor_data = np.array(self.get_distance_sensors_data()) / 20

        state = np.append(position, angle)
        # state = np.append(state, velocity)

        return state

    def step(self, action):
        dt = 0.1

        # 1. 更新领导者位置 (目标动起来)
        self.update_leader(dt)

        # 保存上一时刻的距离，用于计算进度奖励
        prev_distance = self.get_distance()

        self.v = action[0]
        self.w = action[1]

        # 运动学更新
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt

        # 计算当前距离
        curr_distance = self.get_distance()

        # 初始化奖励和结束标志
        reward = 0
        done = False

        # 1. 碰撞惩罚
        if self.has_collided():
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
    print(f"Initial State : {state}")

    action = np.array([100.0, 1])

    # 模拟运行 3 步，观察 Y 值的变化
    for i in range(3):
        next_state, reward, done = env.step(action)
        print(f"Step {i+1} Rep    : x={env.x:.2f}, y={env.y:.2f}, yaw={env.yaw:.2f}")
        print(f"Step {i+1} State  : {next_state}")
