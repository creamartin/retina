import numpy as np 
import matplotlib.pyplot as plt
dev_array=np.zeros(512)
num_dev_array=np.zeros(512)
layer=""
    


def st(auto,man,layer):
    for x in range(512):
        dev=abs(1-(auto[x]/man[x]))*100
        if(np.isfinite(dev) and dev <100):
            dev_array[x]=dev
        num_dev=abs(auto[x]-man[x])
        if(np.isfinite(num_dev) and dev <100):
            num_dev_array[x]=num_dev        
        
    
  
    fig=plt.figure()
    ax2 = fig.add_subplot(222)
    ax2.plot(num_dev_array)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Deviation in pixels')
    

    ax4 = fig.add_subplot(224)
    ax4.plot(dev_array)
    ax4.set_xlabel('x')
    ax4.set_ylabel('Deviation in %')
    

    
                
    fig.suptitle('Pixel deviations: '+layer, fontsize=16)

    t1=("25th percentile : ",round(np.percentile(num_dev_array,25),2),"pixel")
    t2=("50th percentile : ",round(np.percentile(num_dev_array,50),2),"pixel")
    t3=("75th percentile : ",round(np.percentile(num_dev_array,75),2),"pixel")
    t4=("Standard deviation: ", round(np.nanstd(num_dev_array),2))
    t5=("Average: ", round(np.nanmean(num_dev_array),2),"pixel")
    
    t6=("25th percentile : ",round(np.percentile(dev_array,25),2),"%")
    t7=("50th percentile : ",round(np.percentile(dev_array,50),2),"%")
    t8=("75th percentile : ",round(np.percentile(dev_array,75),2),"%")
    t9=("Standard deviation: ", round(np.nanstd(dev_array),2))
    t10=("Average: ", round(np.nanmean(dev_array),2),"%")
    plt.gcf().text(0.02, 0.65, t1, fontsize=10)
    plt.gcf().text(0.02, 0.7, t2, fontsize=10)
    plt.gcf().text(0.02, 0.75, t3, fontsize=10)
    plt.gcf().text(0.02, 0.8, t4, fontsize=10)
    plt.gcf().text(0.02, 0.85, t5, fontsize=10)

    plt.gcf().text(0.02, 0.15, t6, fontsize=10)
    plt.gcf().text(0.02, 0.2, t7, fontsize=10)
    plt.gcf().text(0.02, 0.25, t8, fontsize=10)
    plt.gcf().text(0.02, 0.3, t9, fontsize=10)
    plt.gcf().text(0.02, 0.35, t10, fontsize=10)
    plt.show()




