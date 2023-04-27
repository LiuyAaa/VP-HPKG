import re
#Get the number of operands of each instruction
reg = r"%[0-9]+"
class GET_IN_OUT:
    def __init__(self,ins_name,str1):
        self.input = []
        self.output = []
        self.name = ins_name #Name
        self.str1 = str1 
        self.lines = re.findall(reg, self.str1)
        if ins_name == "alloca":
            self.get_alloca()
        elif ins_name == "store":
            self.get_store()
        elif ins_name == "br":
            self.get_br()
        elif ins_name == "load":
            self.get_load()
        elif ins_name == "icmp":
            self.get_icmp()
        elif ins_name == "add":
            self.get_add()
        elif ins_name == "sub":
            self.get_sub()
        elif ins_name == "call":
            self.get_call()
        elif ins_name == "ret":
            self.get_ret()
        elif ins_name == 'bitcast':
            self.get_bitcast()
        elif ins_name == 'getelementptr':
            self.get_getelementptr()
        elif ins_name == 'sext':
            self.get_sext()
        elif ins_name == 'srem':
            self.get_srem()
        elif ins_name == 'mul':
            self.get_mul()
        elif ins_name == 'phi':
            self.get_phi()
        elif ins_name == 'select':
            self.get_select()
        elif ins_name == 'trunc':
            self.get_trunc()
        elif ins_name == 'ashr':
            self.get_ashr()
        elif ins_name == 'and':
            self.get_and()
        elif ins_name == 'fmul':
            self.get_fmul()
        elif ins_name == 'fdiv':
            self.get_fdiv()
        elif ins_name == 'fcmp':
            self.get_fcmp()
        elif ins_name == 'fadd':
            self.get_fadd()        
    #alloca
    def get_alloca(self):
        self.output.append(self.lines[0])
    #store 
    def get_store(self):
        if len(self.lines) == 2:
            self.input.append(self.lines[0])
            self.output.append(self.lines[1])
        else:
            self.output.append(self.lines[0])
    #br 
    def get_br(self):
        if(len(self.lines) == 1):
            self.output.append(self.lines[0])
        else:
            self.input.append(self.lines[0])
            for i in range(1,len(self.lines)):
                self.output.append(self.lines[i])
    #load
    def get_load(self):
        if(len(self.lines) == 1):
            self.output.append(self.lines[0])
        else:
            self.input.append(self.lines[1])
            self.output.append(self.lines[0])
    #icmp
    def get_icmp(self):
        self.output.append(self.lines[0])
        self.input.append(self.lines[1])
    #add 
    def get_add(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])
    #sub
    def get_sub(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])
    #ret
    def get_ret(self):
        if self.lines:
            self.input.append(self.lines[0])
        else:
            self.input.append(0)
    #call
    def get_call(self):
        if self.lines:
            self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])
    #bitcast
    def get_bitcast(self):
        self.output.append(self.lines[0])
        self.input.append(self.lines[1])
    #getelementptr
    def get_getelementptr(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])
    #sext
    def get_sext(self):
        self.output.append(self.lines[0])
        self.input.append(self.lines[1])
    #srem
    def get_srem(self):
        self.output.append(self.lines[0])
        self.input.append(self.lines[1])
    #mul
    def get_mul(self):
        self.output.append(self.lines[0])
        self.input.append(self.lines[1])
    #phi 
    def get_phi(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])
    #select
    def get_select(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])
    #trunc
    def get_trunc(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])
    #ashr
    def get_ashr(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])
    #and
    def get_and(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])  
    #fmul
    def get_fmul(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])  

    #fdiv
    def get_fdiv(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])  

    #fcmp
    def get_fcmp(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])  

    #fadd
    def get_fadd(self):
        self.output.append(self.lines[0])
        for i in range(1,len(self.lines)):
            self.input.append(self.lines[i])      
    #
    def get_input_output(self):
        return self.input, self.output

