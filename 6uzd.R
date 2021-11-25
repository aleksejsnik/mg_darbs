library(dplyr)

df <- read.csv("uzd_data.csv", sep=";")
print(df)

N <- 1000
WACC <- 0.018

revenue.sd <- df$revenue %>% unlist %>% sd
cog.sd <- df$cog %>% unlist %>% sd
salary.sd <- df$salary %>% unlist %>% sd
property.sd <- df$property %>% unlist %>% sd
delivery.sd <- df$delivery %>% unlist %>% sd


rv.mc <- function(rv.t, app.t) {
  vect <- vector(length = N)
  for (i in 1:N) {
    vect[i] <- (rv.t * (1 + app.t + revenue.sd * rnorm(1))) / 1000000 %>% round
  }
  vect
}

rv.2021 <- rv.mc(47989878, 0.019)
rv.2022 <- rv.mc(48901686, 0.02)
rv.2023 <- rv.mc(49879719, 0.078)
rv.2024 <- rv.mc(53770338, 0.078)
rv.2025 <- rv.mc(57964424, 0.078)

rv <- list(rv.2021, rv.2022, rv.2023, rv.2024, rv.2025)

print.plot <- function(l) {
  for (i in 1:length(l)) {
    qqnorm(unlist(l[i]))
    qqline(unlist(l[i]))
    Sys.sleep(2)
  }
}


make.summary <- function(l){
  Vid <- vector(length = length(l))
  Med <- vector(length = length(l))
  Min <- vector(length = length(l))
  Max <- vector(length = length(l))
  Std.dev <- vector(length = length(l))
  for (i in 1:length(l)) {
    Vid[i] <- l[i] %>% unlist %>% mean
    Med[i] <- l[i] %>% unlist %>% median
    Min[i] <- l[i] %>% unlist %>% min
    Max[i] <- l[i] %>% unlist %>% max
    Std.dev[i] <- l[i] %>% unlist %>% sd
  }
  df <- data.frame(Vid, Med, Min, Max, Std.dev, row.names = 
                     c(2021, 2022, 2023, 2024, 2025))
}

np.mc <- function(rv, cg, pl, ni, it) {
  vect <- vector(length = N)
  for (i in 1:N) {
    cog <- rv * (cg + cog.sd * rnorm(1))
    lc <- rv * (0.1 + salary.sd * rnorm(1))
    pr <- rv * (ni + property.sd * rnorm(1))
    tr <- rv * (0.015 + delivery.sd * rnorm(1))
    ebt <- rv - cog - lc - pl - pr - tr + 300000 - rv * 0.1 - it
    vect[i] <- (ebt * (1 - 0.005)) / 1000000
  }
  vect
}

np.2021 <- np.mc(48901686, 0.712, 1015593, 0.037, 129225)
np.2022 <- np.mc(49879719, 0.712, 991257, 0.04, 110725)
np.2023 <- np.mc(53770338, 0.712, 954511, 0.037, 95000)
np.2024 <- np.mc(57964424, 0.712, 938751, 0.041, 83900)
np.2025 <- np.mc(62485649, 0.712, 920416, 0.037, 76500)

np <- list(np.2021, np.2022, np.2023, np.2024, np.2025)

print.plot(rv)
rv.summary <- make.summary(rv)
print(t(rv.summary))

print.plot(np)
np.summary <- make.summary(np)
print(t(np.summary))

###
pv1.mc <- function(r, o) {
  vect <- vector(length = N)
  for (i in 1:N) {
    pv <- vector(length = 5)
    for (j in 1:5) {
      cog <- r[j] * (0.712 + cog.sd * rnorm(1))
      pv[j] <- (r[j] - cog - o[j]) / (1 + WACC) ^ j
    }
    vect[i] <- pv[j] / 1000000 %>% sum
  }
  vect
}
###
pv2.mc <-function(p) {
  vect <- vector(length = N)
  wacc.sd <- 0.1
  for (i in 1:N) {
    w <- WACC + wacc.sd*rnorm(1)
    pv <- vector(length = 5)
    for (j in 1:5) {
      pv[j] <- p[j] / (1 + w) ^ j
    }
    vect[i] <- pv[j] / 1000000 %>% sum
  }
  vect
}
###

rvv <- c(48901686, 49879719, 53770338, 57964424, 62485649)
other <- c(12862420, 13125184, 14328515, 15609263, 16551179)
pv1 <- pv1.mc(rvv, other)

bnp <- c(1221265, 1240174, 1157342, 1084490, 1444687)
pv2 <- pv2.mc(bnp)

st <- "Std.dev ="

qqnorm(pv1)
qqline(pv1)
print(summary(pv1))
print(paste(st, sd(pv1), sep = " "))

qqnorm(pv2)
qqline(pv2)
print(summary(pv2))
print(paste(st, sd(pv2), sep = " "))


