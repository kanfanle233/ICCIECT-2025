#!/usr/bin/env python3
"""
12小球问题 - 纯算法版本
"""

import random

class TwelveBallSolver:
    def __init__(self):
        self.balls = [0] * 12  # 0=正常，1=较重，-1=较轻
        self.bad_ball = None
        self.bad_ball_type = None

    def setup(self, bad_ball_idx=None, bad_ball_type=None):
        """设置问题：随机或指定坏球"""
        if bad_ball_idx is None:
            self.bad_ball = random.randint(0, 11)
        else:
            self.bad_ball = bad_ball_idx

        if bad_ball_type is None:
            self.bad_ball_type = random.choice([1, -1])
        else:
            self.bad_ball_type = bad_ball_type

        self.balls = [0] * 12
        self.balls[self.bad_ball] = self.bad_ball_type

    def weigh(self, left_balls, right_balls):
        """模拟称重：返回 'left_heavy', 'right_heavy', 'balanced'"""
        left_weight = sum(self.balls[b-1] for b in left_balls)
        right_weight = sum(self.balls[b-1] for b in right_balls)

        if left_weight > right_weight:
            return "left_heavy"
        elif left_weight < right_weight:
            return "right_heavy"
        else:
            return "balanced"

    def solve(self):
        """求解12小球问题"""
        print(f"\n{'='*60}")
        print(f"开始求解12小球问题")
        print(f"坏球位置：{self.bad_ball + 1}，类型：{'较重' if self.bad_ball_type == 1 else '较轻'}")
        print(f"{'='*60}\n")

        steps = 0

        # 第一次称重：1-4 vs 5-8
        print("【第一次称重】球 1-4 vs 球 5-8")
        left = [1, 2, 3, 4]
        right = [5, 6, 7, 8]
        result1 = self.weigh(left, right)
        steps += 1
        print(f"  结果：{result1}")

        if result1 == "balanced":
            # 坏球在9-12中
            print(f"\n  → 坏球在球 9-12 中")
            print()

            # 第二次称重：9-11 vs 1-3（已知正常）
            print("【第二次称重】球 9,10,11 vs 球 1,2,3（已知正常）")
            left = [9, 10, 11]
            right = [1, 2, 3]
            result2 = self.weigh(left, right)
            steps += 1
            print(f"  结果：{result2}")

            if result2 == "balanced":
                # 坏球是12
                print(f"\n  → 坏球是球 12")
                print()

                # 第三次称重：12 vs 1（已知正常）
                print("【第三次称重】球 12 vs 球 1（已知正常）")
                left = [12]
                right = [1]
                result3 = self.weigh(left, right)
                steps += 1
                print(f"  结果：{result3}")

                if result3 == "left_heavy":
                    print(f"\n  ✓ 结论：球 12 较重")
                    return 12, "较重", steps
                else:
                    print(f"\n  ✓ 结论：球 12 较轻")
                    return 12, "较轻", steps

            elif result2 == "left_heavy":
                # 坏球在9-11中且较重
                print(f"\n  → 坏球在球 9-11 中且较重")
                print()

                # 第三次称重：9 vs 10
                print("【第三次称重】球 9 vs 球 10")
                left = [9]
                right = [10]
                result3 = self.weigh(left, right)
                steps += 1
                print(f"  结果：{result3}")

                if result3 == "balanced":
                    print(f"\n  ✓ 结论：球 11 较重")
                    return 11, "较重", steps
                elif result3 == "left_heavy":
                    print(f"\n  ✓ 结论：球 9 较重")
                    return 9, "较重", steps
                else:
                    print(f"\n  ✓ 结论：球 10 较重")
                    return 10, "较重", steps

            else:  # right_heavy
                # 坏球在9-11中且较轻
                print(f"\n  → 坏球在球 9-11 中且较轻")
                print()

                # 第三次称重：9 vs 10
                print("【第三次称重】球 9 vs 球 10")
                left = [9]
                right = [10]
                result3 = self.weigh(left, right)
                steps += 1
                print(f"  结果：{result3}")

                if result3 == "balanced":
                    print(f"\n  ✓ 结论：球 11 较轻")
                    return 11, "较轻", steps
                elif result3 == "left_heavy":
                    print(f"\n  ✓ 结论：球 10 较轻")
                    return 10, "较轻", steps
                else:
                    print(f"\n  ✓ 结论：球 9 较轻")
                    return 9, "较轻", steps

        elif result1 == "left_heavy":
            # 坏球在1-8中
            print(f"\n  → 坏球在球 1-8 中")
            print()

            # 第二次称重：1,2,5 vs 3,6,9（已知正常）
            print("【第二次称重】球 1,2,5 vs 球 3,6,9（已知正常）")
            left = [1, 2, 5]
            right = [3, 6, 9]
            result2 = self.weigh(left, right)
            steps += 1
            print(f"  结果：{result2}")

            if result2 == "balanced":
                # 坏球在4,7,8中
                print(f"\n  → 坏球在球 4,7,8 中")
                print()

                # 第三次称重：7 vs 8
                print("【第三次称重】球 7 vs 球 8")
                left = [7]
                right = [8]
                result3 = self.weigh(left, right)
                steps += 1
                print(f"  结果：{result3}")

                if result3 == "balanced":
                    print(f"\n  ✓ 结论：球 4 较重")
                    return 4, "较重", steps
                elif result3 == "left_heavy":
                    print(f"\n  ✓ 结论：球 8 较轻")
                    return 8, "较轻", steps
                else:
                    print(f"\n  ✓ 结论：球 7 较轻")
                    return 7, "较轻", steps

            elif result2 == "left_heavy":
                # 坏球在1,2中（重）或6中（轻）
                print(f"\n  → 坏球在球 1,2（重）或球 6（轻）中")
                print()

                # 第三次称重：1 vs 2
                print("【第三次称重】球 1 vs 球 2")
                left = [1]
                right = [2]
                result3 = self.weigh(left, right)
                steps += 1
                print(f"  结果：{result3}")

                if result3 == "balanced":
                    print(f"\n  ✓ 结论：球 6 较轻")
                    return 6, "较轻", steps
                elif result3 == "left_heavy":
                    print(f"\n  ✓ 结论：球 1 较重")
                    return 1, "较重", steps
                else:
                    print(f"\n  ✓ 结论：球 2 较重")
                    return 2, "较重", steps

            else:  # right_heavy
                # 坏球在5中（轻）或3中（重）
                print(f"\n  → 坏球在球 5（轻）或球 3（重）中")
                print()

                # 第三次称重：5 vs 9（已知正常）
                print("【第三次称重】球 5 vs 球 9（已知正常）")
                left = [5]
                right = [9]
                result3 = self.weigh(left, right)
                steps += 1
                print(f"  结果：{result3}")

                if result3 == "balanced":
                    print(f"\n  ✓ 结论：球 3 较重")
                    return 3, "较重", steps
                else:
                    print(f"\n  ✓ 结论：球 5 较轻")
                    return 5, "较轻", steps

        else:  # right_heavy
            # 坏球在1-8中
            print(f"\n  → 坏球在球 1-8 中")
            print()

            # 第二次称重：1,2,5 vs 3,6,9（已知正常）
            print("【第二次称重】球 1,2,5 vs 球 3,6,9（已知正常）")
            left = [1, 2, 5]
            right = [3, 6, 9]
            result2 = self.weigh(left, right)
            steps += 1
            print(f"  结果：{result2}")

            if result2 == "balanced":
                # 第一次右盘重→球4在左盘可能是轻，球7,8在右盘可能是重
                print(f"\n  → 坏球在球 4（轻）或球 7,8（重）中")
                print()

                # 第三次称重：7 vs 8
                print("【第三次称重】球 7 vs 球 8")
                left = [7]
                right = [8]
                result3 = self.weigh(left, right)
                steps += 1
                print(f"  结果：{result3}")

                if result3 == "balanced":
                    print(f"\n  ✓ 结论：球 4 较轻")
                    return 4, "较轻", steps
                elif result3 == "left_heavy":
                    print(f"\n  ✓ 结论：球 7 较重")
                    return 7, "较重", steps
                else:
                    print(f"\n  ✓ 结论：球 8 较重")
                    return 8, "较重", steps

            elif result2 == "left_heavy":
                # 坏球在5中（重）或3中（轻）
                # 球5在第一次右盘→可能是重；球3在第一次左盘→可能是轻
                print(f"\n  → 坏球在球 5（重）或球 3（轻）中")
                print()

                # 第三次称重：5 vs 9（已知正常）
                print("【第三次称重】球 5 vs 球 9（已知正常）")
                left = [5]
                right = [9]
                result3 = self.weigh(left, right)
                steps += 1
                print(f"  结果：{result3}")

                if result3 == "balanced":
                    print(f"\n  ✓ 结论：球 3 较轻")
                    return 3, "较轻", steps
                elif result3 == "left_heavy":
                    print(f"\n  ✓ 结论：球 5 较重")
                    return 5, "较重", steps
                else:
                    print(f"\n  ✓ 结论：球 5 较轻")
                    return 5, "较轻", steps

            else:  # right_heavy
                # 坏球在1,2中（轻）或6中（重）
                print(f"\n  → 坏球在球 1,2（轻）或球 6（重）中")
                print()

                # 第三次称重：1 vs 2
                print("【第三次称重】球 1 vs 球 2")
                left = [1]
                right = [2]
                result3 = self.weigh(left, right)
                steps += 1
                print(f"  结果：{result3}")

                if result3 == "balanced":
                    print(f"\n  ✓ 结论：球 6 较重")
                    return 6, "较重", steps
                elif result3 == "left_heavy":
                    print(f"\n  ✓ 结论：球 2 较轻")
                    return 2, "较轻", steps
                else:
                    print(f"\n  ✓ 结论：球 1 较轻")
                    return 1, "较轻", steps

    def test_random(self, num_tests=10):
        """随机测试算法"""
        print(f"\n{'='*60}")
        print(f"运行 {num_tests} 次随机测试")
        print(f"{'='*60}")

        for i in range(num_tests):
            self.setup()
            ball, ball_type, steps = self.solve()

            # 验证结果
            assert ball == self.bad_ball + 1, f"错误：期望球 {self.bad_ball + 1}，实际找到球 {ball}"
            expected_type = "较重" if self.bad_ball_type == 1 else "较轻"
            assert ball_type == expected_type, f"错误：期望 {expected_type}，实际 {ball_type}"
            assert steps == 3, f"错误：期望3步，实际{steps}步"

            print(f"\n✓ 测试 {i+1} 通过：球 {ball} {ball_type}（{steps} 步）")

        print(f"\n{'='*60}")
        print(f"所有测试通过！✓")
        print(f"{'='*60}\n")

    def test_all_cases(self):
        """测试所有24种可能情况"""
        print(f"\n{'='*60}")
        print(f"测试所有 24 种可能情况")
        print(f"{'='*60}")

        success_count = 0
        for ball_idx in range(12):
            for ball_type in [1, -1]:
                self.setup(ball_idx, ball_type)
                ball, found_type, steps = self.solve()

                expected_ball = ball_idx + 1
                expected_type = "较重" if ball_type == 1 else "较轻"

                if ball == expected_ball and found_type == expected_type and steps == 3:
                    success_count += 1
                    print(f"✓ 球 {expected_ball} {expected_type}：成功（{steps} 步）")
                else:
                    print(f"✗ 球 {expected_ball} {expected_type}：失败")

        print(f"\n{'='*60}")
        print(f"成功：{success_count}/24")
        print(f"{'='*60}\n")

def main():
    solver = TwelveBallSolver()

    # 演示一次求解
    print("\n" + "="*60)
    print("12小球问题求解演示")
    print("="*60)

    solver.setup()
    ball, ball_type, steps = solver.solve()

    print(f"\n总结：在 {steps} 次称重内找出坏球为球 {ball}，它 {ball_type}")

    # 运行随机测试
    solver.test_random(5)

if __name__ == "__main__":
    main()
