#import numpy as np
#import pandas as pd
#
#
##frequency 1
#f1_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/1_1.txt') 
#f2_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/1_2.txt')
#f3_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/1_3.txt')
#f4_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/1_4.txt')
#f5_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/1_5.txt')
#
#
## select channels and eliminate bad data
#df1_1 = pd.DataFrame(f1_1[200:5000,7:27])
#df2_1=  pd.DataFrame(f2_1[200:5000,7:27])
#df3_1=  pd.DataFrame(f3_1[200:5000,7:27])
#df4_1=  pd.DataFrame(f4_1[200:5000,7:27])
#df5_1=  pd.DataFrame(f5_1[200:5000,7:27])
#
#
#frame_1=[df1_1, df2_1, df3_1, df4_1, df5_1]
#
#
#df1=pd.concat(frame_1,ignore_index=True)
#
##df1_combine_200=df1.ix[0,:]
##
##for i in range(199):  
##    
##    df1_combine_200=pd.concat([df1_combine_200,df1.ix[i+1,:].T],axis=1,ignore_index=True)
#
#label_1=pd.DataFrame(np.ones(len(df1)))
#
#Data1=pd.concat([df1,label_1],axis=1,ignore_index=True)
#
#
##frequency 2
#f1_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/2_1.txt') 
#f2_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/2_2.txt')
#f3_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/2_3.txt')
#f4_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/2_4.txt')
#f5_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/2_5.txt')
## select channels and eliminate bad data
#df1_2 =  pd.DataFrame(f1_2[200:5000,7:27])
#df2_2 =  pd.DataFrame(f2_2[200:5000,7:27])
#df3_2 =  pd.DataFrame(f3_2[200:5000,7:27])
#df4_2 =  pd.DataFrame(f4_2[200:5000,7:27])
#df5_2 =  pd.DataFrame(f5_2[200:5000,7:27])
#
#frame_2=[df1_2, df2_2, df3_2, df4_2, df5_2]
#
#df2=pd.concat(frame_2,ignore_index=True)
#
#label_2=pd.DataFrame(2*np.ones(len(df2)))
#
#Data2=pd.concat([df2,label_2],axis=1,ignore_index=True)
#
#
#
##frequency 3
#f1_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/3_1.txt') 
#f2_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/3_2.txt')
#f3_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/3_3.txt')
#f4_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/3_4.txt')
#f5_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/3_5.txt')
## select channels and eliminate bad data
#df1_3 =  pd.DataFrame(f1_3[200:5000,7:27])
#df2_3 =  pd.DataFrame(f2_3[200:5000,7:27])
#df3_3 =  pd.DataFrame(f3_3[200:5000,7:27])
#df4_3 =  pd.DataFrame(f4_3[200:5000,7:27])
#df5_3 =  pd.DataFrame(f5_3[200:5000,7:27])
#
#frame_3=[df1_3, df2_3, df3_3, df4_3, df5_3]
#
#df3=pd.concat(frame_3,ignore_index=True)
#
#label_3=pd.DataFrame(3*np.ones(len(df3)))
#
#Data3=pd.concat([df3,label_3],axis=1,ignore_index=True)
#
#
#
##frequency 4
#f1_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/4_1.txt') 
#f2_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/4_2.txt')
#f3_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/4_3.txt')
#f4_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/4_4.txt')
#f5_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/4_5.txt')
## select channels and eliminate bad data
#df1_4 =  pd.DataFrame(f1_4[200:5000,7:27])
#df2_4 =  pd.DataFrame(f2_4[200:5000,7:27])
#df3_4 =  pd.DataFrame(f3_4[200:5000,7:27])
#df4_4 =  pd.DataFrame(f4_4[200:5000,7:27])
#df5_4 =  pd.DataFrame(f5_4[200:5000,7:27])
#
#frame_4=[df1_4, df2_4, df3_4, df4_4, df5_4]
#
#df4=pd.concat(frame_4,ignore_index=True)
#
#label_4=pd.DataFrame(4*np.ones(len(df4)))
#
#Data4=pd.concat([df4,label_4],axis=1,ignore_index=True)
#
#
#
#
##data combination
#Data=pd.concat([Data1,Data2,Data3,Data4],ignore_index=True)
#
#np.random.seed()
#
#Data= Data.sample(frac=1).reset_index(drop=True)
#
#Data.to_csv(path_or_buf ='Data_4label_s2.csv')





#regard the last document as test data

import numpy as np
import pandas as pd



n=5  #number of documents on each label, e.g., n=5

#frequency 1
f1_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/1_1.txt')
f2_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/1_2.txt')
f3_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/1_3.txt')
f4_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/1_4.txt')
f5_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/1_5.txt')


# select channels and eliminate bad data
df1_1 = np.array(f1_1[200:5000,7:27])
df2_1=  np.array(f2_1[200:5000,7:27])
df3_1=  np.array(f3_1[200:5000,7:27])
df4_1=  np.array(f4_1[200:5000,7:27])
df5_1=  np.array(f5_1[200:5000,7:27])


#frame_1=[df1_1, df2_1, df3_1, df4_1]


df1=np.concatenate((df1_1, df2_1, df3_1, df4_1, df5_1), axis=0)

df1_blow_200=np.zeros((48*n-3,4020))

for k in range(48*n-3):
    
    df1_combine_200=df1[100*k,:]
    for i in range(200):
        df1_combine_200=np.concatenate((df1_combine_200,df1[100*k+i+1,:].T))
    
    df1_blow_200[k,:]=df1_combine_200



label_1=pd.DataFrame(np.ones(len(df1_blow_200)))

Data1=np.concatenate((df1_blow_200,label_1),axis=1)


    


#frequency 2
f1_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/2_1.txt') 
f2_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/2_2.txt')
f3_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/2_3.txt')
f4_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/2_4.txt')
f5_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/2_5.txt')
# select channels and eliminate bad data
df1_2 =  np.array(f1_2[200:5000,7:27])
df2_2 =  np.array(f2_2[200:5000,7:27])
df3_2 =  np.array(f3_2[200:5000,7:27])
df4_2 =  np.array(f4_2[200:5000,7:27])
df5_2 =  np.array(f5_2[200:5000,7:27])

#frame_2=[df1_2, df2_2, df3_2, df4_2]

df2=np.concatenate((df1_2, df2_2, df3_2, df4_2, df5_2), axis=0)

df2_blow_200=np.zeros((48*n-3,4020))

for k in range(48*n-3):
    
    df2_combine_200=df2[100*k,:]
    
    for i in range(200):
        df2_combine_200=np.concatenate((df2_combine_200,df2[100*k+i+1,:].T))
    
    df2_blow_200[k,:]=df2_combine_200



label_2=pd.DataFrame(2*np.ones(len(df2_blow_200)))

Data2=np.concatenate((df2_blow_200,label_2),axis=1)



#frequency 3
f1_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/3_1.txt') 
f2_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/3_2.txt')
f3_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/3_3.txt')
f4_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/3_4.txt')
f5_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/3_5.txt')
# select channels and eliminate bad data
df1_3 =  np.array(f1_3[200:5000,7:27])
df2_3 =  np.array(f2_3[200:5000,7:27])
df3_3 =  np.array(f3_3[200:5000,7:27])
df4_3 =  np.array(f4_3[200:5000,7:27])
df5_3 =  np.array(f5_3[200:5000,7:27])

df3=np.concatenate((df1_3, df2_3, df3_3, df4_3, df5_3), axis=0)

df3_blow_200=np.zeros((48*n-3,4020))

for k in range(48*n-3):
    
    df3_combine_200=df3[100*k,:]
    for i in range(200):
        df3_combine_200=np.concatenate((df3_combine_200,df3[100*k+i+1,:].T))
    
    df3_blow_200[k,:]=df3_combine_200



label_3=pd.DataFrame(3*np.ones(len(df3_blow_200)))

Data3=np.concatenate((df3_blow_200,label_3),axis=1)


#frequency 4
f1_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/4_1.txt') 
f2_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/4_2.txt')
f3_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/4_3.txt')
f4_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/4_4.txt')
f5_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s2/4_5.txt')
# select channels and eliminate bad data
df1_4 =  np.array(f1_4[200:5000,7:27])
df2_4 =  np.array(f2_4[200:5000,7:27])
df3_4 =  np.array(f3_4[200:5000,7:27])
df4_4 =  np.array(f4_4[200:5000,7:27])
df5_4 =  np.array(f5_4[200:5000,7:27])

df4=np.concatenate((df1_4, df2_4, df3_4, df4_4, df5_4), axis=0)

df4_blow_200=np.zeros((48*n-3,4020))

for k in range(48*n-3):
    
    df4_combine_200=df4[100*k,:]
    for i in range(200):
        df4_combine_200=np.concatenate((df4_combine_200,df4[100*k+i+1,:].T))
    
    df4_blow_200[k,:]=df4_combine_200



label_4=pd.DataFrame(4*np.ones(len(df4_blow_200)))

Data4=np.concatenate((df4_blow_200,label_4),axis=1)




#datga combination
Data_blow=np.concatenate((Data1,Data2,Data3,Data4),axis=0)



Data_blow =  np.random.permutation(Data_blow)

np.savetxt('Data_blow_s1.csv', Data_blow,fmt='%.4e', delimiter=',')

#
#
##test data combination
#f5_1=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/1_5.txt')
#df5_1=  pd.DataFrame(f5_1[200:5000,7:27])
#label_1=pd.DataFrame(np.ones(len(df5_1)))
#Test1=pd.concat([df5_1,label_1],axis=1,ignore_index=True)
#
#f5_2=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/2_5.txt')
#df5_2 =  pd.DataFrame(f5_2[200:5000,7:27])
#label_2=pd.DataFrame(2*np.ones(len(df5_2)))
#Test2=pd.concat([df5_2,label_2],axis=1,ignore_index=True)
#
#f5_3=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/3_5.txt')
#df5_3 =  pd.DataFrame(f5_3[200:5000,7:27])
#label_3=pd.DataFrame(3*np.ones(len(df5_3)))
#Test3=pd.concat([df5_3,label_3],axis=1,ignore_index=True)
#
#f5_4=np.loadtxt('/Users/SongJialin/Desktop/CS 446/Project/s1/4_5.txt')
#df5_4 =  pd.DataFrame(f5_4[200:5000,7:27])
#label_4=pd.DataFrame(4*np.ones(len(df5_4)))
#Test4=pd.concat([df5_4,label_4],axis=1,ignore_index=True)
#
#Data_test=pd.concat([Test1,Test2,Test3,Test4],ignore_index=True)
#
#np.random.seed()

#Data_test= Data_test.sample(frac=1).reset_index(drop=True)
#
#Data_test.to_csv(path_or_buf ='Data_test_s1.csv')
