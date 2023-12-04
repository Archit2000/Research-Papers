

dataset = Planetoid(root='data/planetoid', split="random",num_val= 1055, num_test= 1055, name=dataset_global)
print(dataset[0])

edge_list = dataset[0].edge_index
NO_OF_EDGES = edge_list.shape[1]
labels = dataset[0].y

print("Homophilic ratio : " + str(homophily(edge_list,labels,method='edge')))


adj = to_dense_adj(dataset[0].edge_index)
adj = adj[0]

labels = labels.numpy()

X = dataset[0].x
X = X.to_dense()
N = X.shape[0]
NO_OF_CLASSES = len(set(labels))

sparsity_original = 2*NO_OF_EDGES/(N*(N-1))
print("Sparsity of original graph : " + str(sparsity_original))

nn = int(1*N)
X = X[:nn,:]
adj = adj[:nn,:nn]
labels = labels[:nn]
print(X.shape,adj.shape)


# In[29]:


def get_laplacian(adj):
    b=torch.ones(adj.shape[0])
    return torch.diag(adj@b)-adj

theta = get_laplacian(adj)
print(theta.shape)


# In[30]:


features = X.numpy()
NO_OF_NODES = X.shape[0]
print(NO_OF_CLASSES,NO_OF_NODES)


# In[31]:


def convertScipyToTensor(coo):
    try:
        coo = coo.tocoo()
    except:
        coo = coo
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


# In[32]:


from scipy.sparse import csr_matrix
from scipy.sparse import random
from scipy.sparse.linalg import norm
from scipy.stats import rv_continuous

p = X.shape[0]
k = int(p*r_global)
k_global = int(p*r_global)
n = X.shape[1]
lambda_param = 100
beta_param = 50
alpha_param = 100
gamma_param = 100
lr = 1e-5
thresh = 1e-10


class CustomDistribution(rv_continuous):
    def _rvs(self,  size=None, random_state=None):
        return random_state.standard_normal(size)
temp = CustomDistribution(seed=1)
temp2 = temp()  # get a frozen version of the distribution
X_tilde = random(k, n, density=0.25, random_state=1, data_rvs=temp2.rvs)
C = random(p, k, density=0.25, random_state=1, data_rvs=temp2.rvs)


# In[33]:


def experiment(alpha_param,beta_param,gamma_param,lambda_param,delta_param,C,X_tilde,theta,X):
    p = X.shape[0]
    k = int(p*r_global)
    n = X.shape[1]
    ones = csr_matrix(np.ones((k,k)))
    ones = convertScipyToTensor(ones)
    ones = ones.to_dense()
    J = np.outer(np.ones(k), np.ones(k))/k
    J = csr_matrix(J)
    J = convertScipyToTensor(J)
    J = J.to_dense()
    zeros = csr_matrix(np.zeros((p,k)))
    zeros = convertScipyToTensor(zeros)
    zeros = zeros.to_dense()
    X_tilde = convertScipyToTensor(X_tilde)
    X_tilde = X_tilde.to_dense()
    C = convertScipyToTensor(C)
    C = C.to_dense()
    eye = torch.eye(k)
    try:
        theta = convertScipyToTensor(theta)
    except:
        theta = theta
    try:
        X = convertScipyToTensor(X)
        X = X.to_dense()
    except:
        X = X

    def one_hot(x, class_count):
        return torch.eye(class_count)[x, :]

    P = labels
    P = one_hot(P,NO_OF_CLASSES)
    
    if(torch.cuda.is_available()):
        # print("yes")
        X_tilde = X_tilde.cuda()
        C = C.cuda()
        theta = theta.cuda()
        X = X.cuda()
        J = J.cuda()
        P = P.cuda()
        zeros = zeros.cuda()
        ones = ones.cuda()
        eye = eye.cuda()
    def update(X_tilde,C,i):
        global L
        thetaC = theta@C
        CT = torch.transpose(C,0,1)
        X_tildeT = torch.transpose(X_tilde,0,1)
        CX_tilde = C@X_tilde
        t1 = CT@thetaC + J
        term_bracket = torch.linalg.pinv(t1)
        thetacX_tilde = thetaC@(X_tilde)
        
        L = 1/k

        t1 = -2*gamma_param*(thetaC@term_bracket)
        t2 = alpha_param*(CX_tilde-X)@(X_tildeT)
        t3 = 2*thetacX_tilde@(X_tildeT)
        t4 = lambda_param*(C@ones)
        t5 = 2*beta_param*(thetaC@CT@thetaC)
        t6 = delta_param*P@torch.transpose((CT@P),0,1)
        T2 = (t1+t2+t3+t4+t5+t6)/L
        Cnew = (C-T2).maximum(zeros)
        t1 = CT@thetaC*(2/alpha_param)
        t2 = CT@C
        t1 = torch.linalg.pinv(t1+t2)
        t1 = t1@CT
        t1 = t1@X
        X_tilde_new = t1
        Cnew[Cnew<thresh] = thresh
        for i in range(len(Cnew)):
            Cnew[i] = Cnew[i]/torch.linalg.norm(Cnew[i],1)
        for i in range(len(X_tilde_new)):
            X_tilde_new[i] = X_tilde_new[i]/torch.linalg.norm(X_tilde_new[i],1)
        return X_tilde_new,Cnew


    for i in tqdm(range(20)):
        X_tilde,C = update(X_tilde,C,i)
    
    return X_tilde,C


# In[34]:


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(X.shape[1], 64)
        self.conv2 = GCNConv(64, NO_OF_CLASSES)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        #print("Checking 1: x", x.shape, "Edge index:", edge_index.shape)
        x = self.conv1(x, edge_index)
        #print("Checking 2: convolution done, new x:", x.shape)
        x = F.relu(x)
        #print("Checking 3: x", x.shape, "training:", self.training)
        x = F.dropout(x, training=self.training)
        #print("Checking 4: dropout done new x", x.shape, "Edge index:", edge_index.shape)
        x = self.conv2(x, edge_index)
        #print("Checking 5: x", x.shape)

        return F.log_softmax(x, dim=1)


# In[35]:


from random import sample
from torch_geometric.utils import dense_to_sparse,homophily
def get_accuracy(C_0,L,X_t_0):
    global labels, NO_OF_CLASSES,k
    t=[]
    for i in [1,2,3,4,5,6,7,8,9,10]: 
        C_0_new=np.zeros(C_0.shape)
        for i in range(C_0.shape[0]):
            C_0_new[i][np.argmax(C_0[i])]=1
        # print(C_0_new)
        # C_0_new=C_0
        from scipy import sparse
        #Lc=C_0.T@L@C_0
        Lc=C_0_new.T@L@C_0_new
        # print("L:", Lc.shape)
        # Lc=L_new
        #print(Lc)
        Wc=(-1*Lc)*(1-np.eye(Lc.shape[0]))
        # print("W:", Wc.shape)
        Wc[Wc<0.1]=0
        Wc=sparse.csr_matrix(Wc)
        Wc = Wc.tocoo()
        row = torch.from_numpy(Wc.row).to(torch.long)
        col = torch.from_numpy(Wc.col).to(torch.long)
        edge_index_coarsen2 = torch.stack([row, col], dim=0)
        #print("edgecoarsen:", edge_index_coarsen2.shape)
        edge_weight = torch.from_numpy(Wc.data)
        #print("edgeweight:", edge_weight.shape)
        def one_hot(x, class_count):
            return torch.eye(class_count)[x, :]

        device = torch.device('cpu')
        labels=labels
        Y = labels
        #print("Y:", Y.shape)
        Y = one_hot(Y,NO_OF_CLASSES)
        # NO_OF_CLASSES=Y.shape[1]
        P=np.linalg.pinv(C_0_new)
        labels_coarse = torch.argmax(torch.sparse.mm(torch.Tensor(P).double() , Y.double()).double() , 1)
        #print("Lables:", labels_coarse.shape)

        #torch.Tensor(C2)@X
        # Wc[Wc<0.01]=0
        Wc=Wc.toarray()
        adjtemp = torch.tensor(Wc)
        edge_list_temp = dense_to_sparse(adjtemp)[0]
        # print(edge_list_temp)
        # print(labels_coarse)
        print("Homophilic ratio : " + str(homophily(edge_list_temp,labels_coarse,method='edge')))
        number_of_edges = edge_list_temp.shape[1]
        n = labels_coarse.shape[0]
        sparsity = 2*number_of_edges/(n*(n-1))
        print("Sparsity : " + str(sparsity))
    
        #
        C2=np.linalg.pinv(C_0_new)
        model=Net().to(device)
        device = torch.device('cpu')
        lr=0.01
        decay=0.0001
        try:
            X=np.array(features.todense())
        except:
            X = np.array(features)
        #print("X:",X.shape)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        # criterion=torch.nn.CrossEntropyLoss()
        x=sample(range(0, int(k)), k)
      
        from datetime import datetime
        Xt=P@X
        # Xt=X_t_0
        def train():
            model.train()
            optimizer.zero_grad()
            out = model(torch.Tensor(Xt).to(device),edge_index_coarsen2)
            loss = F.nll_loss(out[x], labels_coarse[x])
            loss.backward()
            optimizer.step()
            return loss
        
        def test():
            model.eval()
            out = model(dataset[0].x, dataset[0].edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[dataset[0].test_mask] == dataset[0].y[dataset[0].test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(dataset[0].test_mask.sum())  # Derive ratio of correct predictions.
            return test_acc

        def val():
            model.eval()
            out = model(dataset[0].x, dataset[0].edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            val_correct = pred[dataset[0].val_mask] == dataset[0].y[dataset[0].val_mask]  # Check against ground-truth labels.
            val_acc = int(val_correct.sum()) / int(dataset[0].val_mask.sum())  # Derive ratio of correct predictions.
            return val_acc
        

        now1 = datetime.now()
        losses=[]
        for epoch in range(200):
            loss=train()
            losses.append(loss)
            if(epoch%100==0):
                print(f'Epoch: {epoch:03d},loss: {loss:.4f}')
        now2 = datetime.now()        
        pred=model(torch.Tensor(Xt).to(device),edge_index_coarsen2).argmax(dim=1)        
        def train_accuracy():
            model.eval()
            correct = (pred[x] == labels_coarse[x]).sum()
            acc = int(correct) /len(x)
            return acc
    
        t+=[(now2-now1).total_seconds()]

        zz=sample(range(0, int(NO_OF_NODES)), NO_OF_NODES)
        Wc=sparse.csr_matrix(adj)
        Wc = Wc.tocoo()
        row = torch.from_numpy(Wc.row).to(torch.long)
        col = torch.from_numpy(Wc.col).to(torch.long)
        edge_index_coarsen = torch.stack([row, col], dim=0)
        edge_weight = torch.from_numpy(Wc.data)
        pred=model(torch.Tensor(X),edge_index_coarsen).argmax(dim=1)
        pred=np.array(pred)
        correct =(pred[zz]==labels[zz]).sum()
        acc = int(correct) /NO_OF_NODES
        test_acc = test()
        val_acc = val()
        print(f'Train Accuracy: {acc:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Validtion Accuracy: {val_acc:.4f}')
        return acc


# In[36]:


def getSparsityAndHomophily(C,theta):
    theta = C.T@theta@C
    adjtemp = -theta
    for i in range(adjtemp.shape[0]):
        adjtemp[i,i]=0
    adjtemp[adjtemp<0.01]=0
    temp = dense_to_sparse(adjtemp)
    edge_list_temp = temp[0]
    # ytemp = temp[1]
    # P = torch.linalg.pinv(C)
    # labels = 
    # # print(edge_list)
    number_of_edges = edge_list_temp.shape[1]
    # n = adjtemp.shape[0]

    # print("Homophilic ratio : " + str(homophily(edge_list_temp,ytemp,method='node')))
    sparsity = 2*number_of_edges/(n*(n-1))
    print("Sparsity : " + str(sparsity))


# In[37]:


def fitness_function(alpha_param,beta_param,gamma_param,lambda_param,delta_param):
    print("\n---------------------------------------------------------------------------------------------------------------")
    print(alpha_param,beta_param,gamma_param,lambda_param,delta_param)
    try:
        #alpha_param,beta_param,gamma_param,lambda_param,delta_param = temp_param
        X_tilde = random(k, n, density=0.15, random_state=1, data_rvs=temp2.rvs)
        C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
        X_t_0,C_0 = experiment(alpha_param,beta_param,gamma_param,lambda_param,delta_param,C,X_tilde,theta,X)
        L = theta
        
        #getSparsityAndHomophily(C_0,theta)
        
        C_0 = C_0.cpu().detach().numpy()
        X_t_0 = X_t_0.cpu().detach().numpy()
        C_t_0 = C_0.T

        try:
            L = L.cpu().detach().numpy()
        except:
            L = L
        
        acc = get_accuracy(C_0,L,X_t_0)
        #print("Accuracy = " + str(acc))
        #save_readings(alpha_param,beta_param,gamma_param,lambda_param,delta_param,r_global,k_global,acc)
        #return (1-acc)
    except Exception as e:
        print(e)

for alpha_param in [0.001]:
    for beta_param in [0.001]:
        for gamma_param in [100]:
            for lambda_param in [100]:
                for delta_param in [0.001]:
                    try:
                        fitness_function(alpha_param,beta_param,gamma_param,lambda_param,delta_param)
                    except Exception as e:
                        print(e)
# In[38]:


lb = [0.0001,0.0001,0.0001,0.0001,0.0001]
ub = [100,100,100,100,100]


# In[39]:


#pso(fitness_function, lb, ub,phip=0.5,swarmsize=150,maxiter= 100,phig=0.45,omega=0.5,debug=True)

