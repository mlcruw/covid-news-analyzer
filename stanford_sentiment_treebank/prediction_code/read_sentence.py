

def main():
    f = open('../../oldgit/covid_19_articles.sentences', 'r')

    while True:
        sentence_num = f.readline()
        sentences = []
        for rows in range(int(sentence_num)):
        	sentence = f.readline()
        	br = f.readline()
        	#print(sentence)
        	#print(br)
        	sentences.append(sentence)
        print(sentences)

        br = f.readline()
        print('\n\n')
        if sentence_num == '-1\n':
        	break
    f.close()

main()