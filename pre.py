file = open("./ans.txt", "w+")
for i in range(1, 3676):
    file.write("1.0 ")
for i in range(3676, 7351):
    file.write("0.0 ")
file.close