# ===== 1. 读取数据（更稳的方式）=====
titanic <- read.csv("数据可视化/R语言/titanic.csv", stringsAsFactors = FALSE)

# ===== 2. 加载包 =====
library(ggplot2)

# ===== 3. 基础检查 =====
str(titanic)
summary(titanic)

# ===== 4. 数据预处理 =====
# 转为因子（更适合画图）
titanic$sex <- as.factor(titanic$sex)
titanic$alive <- as.factor(titanic$alive)
titanic$class <- as.factor(titanic$class)

# 去掉年龄缺失
titanic_age <- titanic[!is.na(titanic$age), ]

# ===== 5. 可视化 =====

# 1）性别 vs 生存
ggplot(titanic, aes(x = sex, fill = alive)) +
  geom_bar(position = "dodge") +
  labs(title = "不同性别的生存情况",
       x = "性别",
       y = "人数",
       fill = "是否生存") +
  scale_fill_manual(values = c("red", "green")) +
  theme_minimal()

# 2）舱位 vs 生存比例（推荐重点图）
ggplot(titanic, aes(x = class, fill = alive)) +
  geom_bar(position = "fill") +
  labs(title = "不同舱位的生存比例",
       x = "舱位",
       y = "比例",
       fill = "是否生存") +
  scale_fill_manual(values = c("red", "green")) +
  theme_minimal()

# 3）年龄分布（去NA）
ggplot(titanic_age, aes(x = age, fill = alive)) +
  geom_histogram(binwidth = 5, alpha = 0.5, position = "identity") +
  labs(title = "年龄分布与生存情况",
       x = "年龄",
       y = "人数",
       fill = "是否生存") +
  theme_minimal()

# 4）票价 vs 年龄（更清晰）
ggplot(titanic_age, aes(x = age, y = fare, color = alive)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "年龄与票价的关系（含趋势线）",
       x = "年龄",
       y = "票价",
       color = "是否生存") +
  theme_minimal()

# 5）性别 + 舱位（高价值图）
ggplot(titanic, aes(x = class, fill = alive)) +
  geom_bar(position = "fill") +
  facet_wrap(~ sex) +
  labs(title = "性别与舱位对生存率的影响",
       x = "舱位",
       y = "生存比例",
       fill = "是否生存") +
  scale_fill_manual(values = c("red", "green")) +
  theme_minimal()

ggplot(titanic, aes(x = class, y = age, fill = alive)) +
  geom_violin(trim = FALSE) +
  labs(title = "不同舱位年龄分布",
       x = "舱位",
       y = "年龄") +
  theme_minimal()

ggplot(titanic, aes(x = age, fill = alive)) +
  geom_histogram(binwidth = 5, alpha = 0.6) +
  facet_grid(sex ~ class) +
  theme_minimal() +
  labs(title = "性别+舱位+年龄+生存的综合分析")