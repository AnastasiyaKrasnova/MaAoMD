from data import *
from Potentials import PotentialMethod

def main():
    classifier=PotentialMethod(feature_data)
    classifier.train()

if __name__ == "__main__":
    main()