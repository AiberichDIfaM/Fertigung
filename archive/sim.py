import classes

a1 = classes.part_type("a1")
a2 = classes.part_type("a2")
a3 = classes.part_type("a3")
a4 = classes.part_type("a4")
a5 = classes.part_type("a5")
a6 = classes.part_type("a6")
a7 = classes.part_type("a7")
a8 = classes.part_type("a8")
a9 = classes.part_type("a9")
a0 = classes.part_type("a0")
b1 = classes.part_type("b1")
b2 = classes.part_type("b2")
b3 = classes.part_type("b3")
b4 = classes.part_type("b4")
b5 = classes.part_type("b5")
b6 = classes.part_type("b6")
b7 = classes.part_type("b7")
b8 = classes.part_type("b8")
b9 = classes.part_type("b9")
b0 = classes.part_type("b0")

tr1 = classes.transformation([a1,a2],a3,3)
tr2 = classes.transformation([a4,a5,a6],a7,6)
tr3 = classes.transformation(a8,a9,2)
tr4 = classes.transformation([a8,a0],b1,2)
tr5 = classes.transformation([a3,a0],b2,3)
tr6 = classes.transformation([b2,a9],b3,5)
tr7 = classes.transformation([b2,a5],b4,5)
tr8 = classes.transformation([a2,a9],b5,5)
tr9 = classes.transformation([b2,a5],b6,5)
tr10 = classes.transformation([b3,b5,b1],b7,5)
tr11 = classes.transformation([b1,a5,a7],b8,5)
tr12 = classes.transformation([b8],b9,5)
tr13 = classes.transformation([b7],b0,5)

fp1_type = classes.part_type("fp1")
fp2_type = classes.part_type("fp2")

ftran1 = classes.transformation([b4,b5,b6,b7],fp1_type,10)
ftran2 = classes.transformation([b1,b2,b3,b9,b8,b0],fp2_type,15)

finalproduct1 = classes.product("fp1",fp1_type,20)
finalproduct2 = classes.product("fp2",fp2_type,30)

m1_type = classes.machine_type("m1",4,[tr1,tr6])
m2_type = classes.machine_type("m2",3,[tr2,tr9])
m3_type = classes.machine_type("m3",2,[tr2,tr5,tr11])
m4_type = classes.machine_type("m4",6,[tr12,tr3,tr7])
m5_type = classes.machine_type("m5",5,[tr4,tr8,ftran1])
m6_type = classes.machine_type("m6",1,[tr4,ftran2])

m1 = classes.machine(m1_type,"m1",[])
m2 = classes.machine(m2_type,"m1",[])
m3 = classes.machine(m3_type,"m1",[])
m4 = classes.machine(m4_type,"m1",[])
m5 = classes.machine(m5_type,"m1",[])
m6 = classes.machine(m6_type,"m1",[])
m8 = classes.machine(m3_type,"m1",[])
m7 = classes.machine(m1_type,"m1",[])
m9 = classes.machine(m4_type,"m1",[])
m0 = classes.machine(m6_type,"m1",[])

array = [m1,m2,m3,m4,m5,m6,m7,m8,m9,m0]
for mchn in array:
    mchn.connected_machines = array.remove(mchn)

input = []

anlage = classes.anlage(array,0,input)

class simulation:
    def __init__(self,preferences):
        self.anlage = anlage
        for i in range(len(preferences)):
            self.anlage.machines[i].input_preferences = preferences[i][0]
            self.anlage.machines[i].output_preferences = preferences[i][1]
    def run(self,condition):
        while not condition:
            self.anlage.refill_buffer()
            for mchn in self.anlage.machines:
                mchn.next_timestep()

print(array)

