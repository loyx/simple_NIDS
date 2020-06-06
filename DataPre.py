import re

def read_file(filename):
    with open(filename, 'r') as f:
        s = f.readlines()
    return s


def prepare_data(data_line: str):
    temp = re.split(',|\n',data_line)
    result = []
    # col 2 1
    if temp[1] == 'icmp':
        result.append('1')
    elif temp[1] == 'tcp':
        result.append('2')
    elif temp[1]=='udp':
        result.append('3')
    else:
        result.append('4')
    # col 3 2
    if temp[2] == 'domain_u':
        result.append('1')
    elif temp[2] == 'ecr_i':
        result.append('2')
    elif temp[2]=='eco_i':
        result.append('3')
    elif temp[2] == 'finger':
        result.append('4')
    elif temp[2] == 'ftp_data':
        result.append('5')
    elif temp[2]=='ftp':
        result.append('6')
    elif temp[2] == 'http':
        result.append('7')
    elif temp[2] == 'hostnames':
        result.append('8')
    elif temp[2] == 'imap4':
        result.append('9')
    elif temp[2] == 'login':
        result.append('10')
    elif temp[2] == 'mtp':
        result.append('11')
    elif temp[2] == 'netstat':
        result.append('12')
    elif temp[2] == 'other':
        result.append('13')
    elif temp[2] == 'private':
        result.append('14')
    elif temp[2] == 'smtp':
        result.append('15')
    elif temp[2] == 'systat':
        result.append('16')
    elif temp[2] == 'telnet':
        result.append('17')
    elif temp[2] == 'time':
        result.append('18')
    elif temp[2] == 'uucp':
        result.append('19')
    else:
        result.append('20')

    # col 4 3
    if temp[3]=='REJ':
        result.append('1')
    elif temp[3]=='RSTO':
        result.append('2')
    elif temp[3]=='RSTR':
        result.append('3')
    elif temp[3]=='S0':
        result.append('4')
    elif temp[3]=='S3':
        result.append('5')
    elif temp[3]=='SF':
        result.append('6')
    elif temp[3] == 'SH':
        result.append('7')
    else:
        result.append('8')

    # col 5 4
    result.append(temp[4])
    # col 6 5
    result.append(temp[5])
    # col 11 6
    result.append(temp[10])
    # col 16 7
    result.append(temp[15])
    # col 23 8
    result.append(temp[22])
    # col 24 9
    result.append(temp[23])
    # col 27 10
    result.append(str(int(int(float(temp[26])*1000)/10)))
    # col 29 11
    # result.append(str(int(float(temp[28]) * 100)))
    result.append(str(int(int(float(temp[28])*1000)/10)))
    # col 30 12
    # result.append(str(int(float(temp[29])*100)))
    result.append(str(int(int(float(temp[29])*1000)/10)))
    # col 33 13
    result.append(temp[32])
    # col 34 14
    # result.append(str(int(float(temp[33])*100)))
    result.append(str(int(int(float(temp[33])*1000)/10)))
    # col 35 15
    # result.append(str(int(float(temp[34])*100)))
    result.append(str(int(int(float(temp[34])*1000)/10)))
    # col 36 16
    # result.append(str(int(float(temp[35])*100)))
    result.append(str(int(int(float(temp[35])*1000)/10)))
    # col 37 17
    # result.append(str(int(float(temp[36])*100)))
    result.append(str(int(int(float(temp[36])*1000)/10)))
    # col 39 18
    # result.append(str(int(float(temp[38])*100)))
    result.append(str(int(int(float(temp[38])*1000)/10)))
    # col 42 19
    if temp[41]=='normal.':
        result.append('0')
    elif temp[41] == 'smurf.' or temp[41]=='neptune.' or temp[41]=='back.' or temp[41]=='teardrop.' or temp[41]=='pod.' or temp[41]=='land.':
        result.append('1')
    elif temp[41] == 'satan.' or temp[41]=='ipsweep.' or temp[41]=='portsweep.' or temp[41]=='nmap.':
        result.append('2')
    elif temp[41] == "buffer_overflow." or temp[41]=="rootkit." or temp[41]=="loadmodule." or temp[41]=='perl.':
        result.append('3')
    elif temp[41] == "ftp_write." or temp[41] == "guess_passwd." or temp[41] == "imap." or temp[41] == "multihop."\
    or temp[41]=="phf." or temp[41]=="spy." or temp[41]=="warezclient." or temp[41]=="warezmaster.":
        result.append('4')
    else:
        result.append('0')
    
    return result
        
def bar(percent=0,all=100,width=30):
    left = width * percent // all
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent*100/all:.0f}%',
          sep='', end='', flush=True)

def main(filename):
    # filename = 'corrected'
    data = read_file(filename)
    percent = 0
    total_number = len(data)
    with open(filename + '_ok', 'w') as f:
        for data_line in data:
            bar(percent, total_number - 1, 60)
            percent+=1
            result = prepare_data(data_line)
            length = len(result)
            for index in range(length):
                if index == length-1:
                    f.write(result[index] + '\n')
                else:
                    f.write(result[index] + ',')

if __name__ == "__main__":
    print("训练文件预处理中...")
    main('kddcup.data_10_percent')
    print()
    print("训练文件预处理结束")
    print("测试文件预处理中...")
    main('corrected')
    print()
    print("测试文件预处理结束")