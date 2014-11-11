# from SDSFoundations or https://courses.edx.org/c4x/UTAustinX/UT.7.01x/asset/BikeData.csv
BikeData <- BikdData 
View(BikeData)

# What is the age of the 7th rider in the data set?
BikeData[7, ]$age
# How many of the first 10 riders in the dataset ride daily?
nrow((BikeData[1:10, ])[BikeData[1:10, ]$cyc_freq == 'Daily', ])

# How many of the cyclists were students?
stu <- BikeData[BikeData$student == 1, ]
nrow(stu)

# How often did they ride?
table(stu$cyc_freq)

# What was the average distance they rode?
stu_dist <- stu$distance
mean(stu_dist)
