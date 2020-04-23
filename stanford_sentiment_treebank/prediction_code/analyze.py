import csv

f = open("covid_19_articles.sentences", "w")


with open('covid_19_articles.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0


    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
        
       
        print("This line: ",row[3])
        s = row[3].splitlines()
        print(len(s), " sentences.")
        sentences = []
        for i in range(len(s)):
            if s[i] == '':
                continue
            print(s[i])
            s[i] = s[i].replace('“', '')
            s[i] = s[i].replace('”', '')
            s[i] = s[i].replace('"', '')

            
            sentences.append(s[i])
            print("============================--------------------======================\n")
        print("Separating lines: ", s)
        print("======")
        if line_count > 1:
            f.write(str(len(sentences)))
            f.write('\n')
            for i in range(len(sentences)):
                f.write(sentences[i])
                f.write('\n')
                f.write('\n')
            f.write('\n')
    print(f'Processed {line_count} lines.')
    print(csv_reader.line_num)
    f.write('-1\n')
f.close()