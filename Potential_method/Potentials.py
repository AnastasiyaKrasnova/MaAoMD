

class PotentialMethod():
    def __init__(self, learning_set):

        self.learning_set = learning_set
        
    def train(self):
        coef=1
        prev_params=[0,0,0,0]
        for i in range(0,len(self.learning_set)):
            category, curr_point, new_point = self.get_points(i)
            curr_params = self.get_params(curr_point, coef)
            prev_params=self.sum_params(prev_params, curr_params)
            K=self.private_potential(prev_params,new_point)
            coef=self.get_coef(K, category)
        self.classification_func_params=prev_params
        return prev_params

    def guess(self,x,y):
        func=self.classification_func_params[0]+self.classification_func_params[1]*x+self.classification_func_params[2]*y+self.classification_func_params[3]*x*y
        if (func>=0):
            return 1
        else:
           return 2
    

    def private_potential(self, params, point):
            return params[0]+params[1]*point[0]+params[2]*point[1]+params[3]*point[0]*point[1]

    def get_params(self,curr, coef):
        return [1*coef, 4*curr[0]*coef, 4*curr[1]*coef, 16*curr[0]*curr[1]*coef]
    
    def sum_params(self, prev, curr):
        res=[]
        for i in range(0,4):
            res.append(prev[i]+curr[i])
        return res

    def get_points(self,i):

        category, curr = self.learning_set[i]
        if (i==len(self.learning_set)-1):
            category, new=self.learning_set[0]
        else:
            category, new=self.learning_set[i+1]
        return category, curr, new

    
    def get_coef(self, K, category):
        if (K<=0 and category==1):
            return 1
        if (K>0 and category==2):
            return -1
        else:
            return 0


