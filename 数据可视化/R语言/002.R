library(scatterplot3d)

data(iris)

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