import torch as T
from config import Hyper, Constants

# It all starts here
def main():
    print("\n"*10)
    print("-"*100)
    print("Start of Person Detection in Images")
    Hyper.display()
    print("-"*100)


    print("\n"*5)  
    print("-"*100)
    Hyper.display()
    print("End of Person Detection in Images")
    print("-"*100)
    
if __name__ == "__main__":
    main()