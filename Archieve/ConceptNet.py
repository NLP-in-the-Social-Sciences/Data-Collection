import requests

def example():

    for number in range(30):
        print("file no: " + str(number+1))

        f=open("ChainHeadHead/news" + str(number+1) + ".txt", "r")

        print()
        outF = open("CNFeatureBoolean/news" + str(number+1) + ".txt", "w")

        if f.mode == 'r':
            lines = f.readlines()
            n=1
            for i in lines:
                v = 0
                obj = requests.get('http://api.conceptnet.io/c/en/' + i.lower(),timeout=1000 ).json()
                print('chain no' + str(n))
                n = n + 1
                for j in obj['edges']:
                    if ('person' in j['@id']):
                        outF.write('1')
                        outF.write("\n")
                        print('1')
                        v=1
                        break
                if v==0:
                    outF.write('0')
                    outF.write("\n")
                    print('0')

        f.close()
        outF.close()

def main():
    # print("Hello World!")
    example()


if __name__ == "__main__":
    main()





