x <- c(2,5,1,3,4,1,5,3,4,2)
x

y <- c(50,57,41,51,54,38,63,48,59,46)
y

plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="l")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="b")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="o")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="s")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="h")

par(mfrow = c(3, 3))

plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="p")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="l")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="o")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="b")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="h")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="s")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="s",col = "red")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="s",col="red",col.axis="blue")
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="s",col="red",col.axis="blue",main="我的分析图",col.main="purple")

plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="s",col="red",col.axis="blue",main="我的分析图",col.main="purple",cex=2,cex.main=3)
plot(x, y, xlab="广告投入（万元）", ylab="广告投入与销售额的关系", type="s",col="red",col.axis="blue",main="我的分析图",col.main="purple",cex=2,cex.main=3)

plot(x,type='b',main='点连线')
plot(x,type='o',main='线穿过点')
plot(x,type='h',main='悬垂线')
plot(x,type='s',main='阶梯线')
plot(1:25,pch=1:25,cex=2,bg="blue",main="PCH自带的符号",xlab = "pch编码")


x=c(2,4,6,8,10)
x=ts(x,start = c(2020,1))
plot(x,type='s',main='阶梯线')
plot(x,type='h',main='阶梯线')

sale1 <- c(10, 12, 15, 20, 18)
sale2 <- c(12, 11, 15, 24, 25)
par(mfrow = c(3, 3))
plot(sale1, type="o", col="red", main="销售趋势图")
lines(sale2, type="o", col="blue")

curve(sin(x),-2*pi,2*pi,type="o")
curve(cos(x),-2*pi,2*pi,type="h")
curve(tan(x),-2*pi,2*pi,type="h")
curve(1/tan(x),-2*pi,2*pi,type="h")

h <- c(144,166,163,143,152,169,130,159,160,175,161,170,146,159,150,183,165,146,169)

barplot(h,
        main = "销售量柱状图",
        xlab = "样本编号",
        ylab = "销售量",
        col = "blue",
        border = "black")

iris$Sepal.Length
plot(density(iris$Sepal.Length),main="Sepal.Length密度图")
# 加载数据
data(iris)

# 设置画布
par(mfrow = c(2, 2))

# 1. 直方图 + 密度曲线
hist(iris$Sepal.Length,
     col = "lightblue",
     main = "Sepal Length 分布",
     xlab = "Sepal Length",
     probability = TRUE)

lines(density(iris$Sepal.Length), col = "red", lwd = 2)

# 2. 按物种箱线图
boxplot(Sepal.Length ~ Species,
        data = iris,
        col = c("red", "green", "blue"),
        main = "不同物种的Sepal Length",
        xlab = "Species",
        ylab = "Sepal Length")

# 3. 散点图 + 回归线
plot(iris$Sepal.Width, iris$Sepal.Length,
     col = as.numeric(iris$Species),
     pch = 19,
     main = "Sepal Width vs Length",
     xlab = "Sepal Width",
     ylab = "Sepal Length")

abline(lm(Sepal.Length ~ Sepal.Width, data = iris),
       col = "black",
       lwd = 2)

# 4. 密度分组对比
plot(density(iris$Sepal.Length[iris$Species=="setosa"]),
     col = "red",
     lwd = 2,
     main = "不同物种密度对比",
     xlab = "Sepal Length")

lines(density(iris$Sepal.Length[iris$Species=="versicolor"]),
      col = "green", lwd = 2)

lines(density(iris$Sepal.Length[iris$Species=="virginica"]),
      col = "blue", lwd = 2)

legend("topright",
       legend = c("setosa", "versicolor", "virginica"),
       col = c("red", "green", "blue"),
       lwd = 2)

data(iris)

# 离散化 Sepal.Length
iris$LengthGroup <- cut(iris$Sepal.Length,
                        breaks = 3,
                        labels = c("短", "中", "长"))

# 构造列联表
tab <- table(iris$LengthGroup, iris$Species)

# 画马赛克图
mosaicplot(tab,
           col = c("red", "green", "blue"),
           main = "Sepal.Length 与 Species 的关系",
           xlab = "Sepal Length 分组",
           ylab = "Species")
mosaicplot(~ LengthGroup + Species,
           data = iris,
           col = c("red", "green", "blue"),
           main = "马赛克图：长度 vs 物种",
           shade = TRUE)

install.packages("scatterplot3d")

library(scatterplot3d)

colors <- c("red", "green", "blue")[as.numeric(iris$Species)]

scatterplot3d(iris$Sepal.Length,
              iris$Sepal.Width,
              iris$Petal.Length,
              color = colors,
              pch = 16,
              angle = 45,
              main = "Iris 3D Visualization")

legend("topright",
       legend = levels(iris$Species),
       col = c("red", "green", "blue"),
       pch = 16)

