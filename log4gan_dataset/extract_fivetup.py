import os
import json


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def sumOfList(list, size):
   if (size == 0):
     return 0
   else:
     return list[size - 1] + sumOfList(list, size - 1)

def addr2int(addr):
    # 将点分十进制IP地址转换成十进制整数
    items = [int(x) if is_number(x) else 'F' for x in addr.split('.')]
    # for x in addr.split('.'):
    #     if is_number(x):
    #         int(x)
    #     else:
    #         return False
    if items[0] == 'F':
        return 65535  # ipv6地址的最大ip可用数
        # return False
    return sumOfList(items, len(items))

def porttype_encode(port):
    if 0 < port < 1023:
        port_dynamic = 1
        port_static = 0
    elif 1024 < port < 65535:
        port_dynamic = 0
        port_static = 1
    else:
        port_dynamic = 0
        port_static = 0
    return port_dynamic, port_static





# def proto2int(proto):
#     # 将协议类型转换为整数
#     result = 0
#     for i in proto:
#         ASCII = ord(i)
#         result = result + ASCII
#
#     return result

def save2txt(json, savepath):
    with open(savepath, "a", encoding="utf-8") as fc:
        fc.write(json+'\n')
        fc.close()



if __name__ == '__main__':
    # 数据路径
    path = r'D:\code\log4gan_dataset\zeeklog\22.9.19\combine_connlogs.log'
    savepath = r"D:\code\log4gan_dataset\zeeklog\22.9.19\conn_with_ID.json"
    # path = input("目标日志文件绝对路径：")
    five_tup_list = []
    access2fivetup_dict = {}
    count = 0
    with open(path, 'r', encoding='utf8')as f:
        for line in f:
            line2dict = json.loads(line)
            # 获取五元组
            src_ip = json.loads(str(line))['id.orig_h']
            dst_ip = json.loads(str(line))['id.resp_h']
            src_port = json.loads(str(line))['id.orig_p']  # int
            dst_port = json.loads(str(line))['id.resp_p']  # int
            proto = json.loads(str(line))['proto']

            # 将源IP、目的IP、目的端口以及协议类型转换为整数,即访问关系ID
            intsrc_ip = addr2int(src_ip)

            # 将目的端口按动(0-1023)/静(1024-65535)态进行分类编码
            port_dynamic, port_static = porttype_encode(dst_port)
            line2dict['port_dynamic'] = port_dynamic
            line2dict['port_static'] = port_static



            # # 忽略ipv6地址
            # if intsrc_ip == False:
            #     continue
            intdst_ip = addr2int(dst_ip)
            # 在计算access_RSID时，有可能会遇到那种比较长的端口，导致计算结果偏大
            # 如果遇到ipv6地址直接使access_RSID为65535,因为其地址长度为ipv4的4倍，按最大值(FFFF,即65535)算
            if intsrc_ip == intdst_ip == 65535:
                access_RSID = 65535
                print("The address is an IPv6 address")
            else:
                access_RSID = intsrc_ip + intdst_ip + dst_port
            # 因为会有部分网络流的访问关系再计算后大小超过65535
            # 故将access_rsid缩放至0~65535的范围内
            # 这样不会影响ipv6地址的访问关系值，因为其缩放后仍为65535
            low4scaling, upper4scaling = 0, 65535
            access_rsid65535 = access_RSID % (upper4scaling-low4scaling+1)+low4scaling

            line2dict["access_RSID"] = access_rsid65535
            del line2dict['ts']
            del line2dict['uid']
            del line2dict['id.orig_p']
            if 'history' in line2dict.keys():
                del line2dict['history']
            del line2dict['missed_bytes']


            print(line2dict)
            save2txt(json.dumps(line2dict), savepath=savepath)


            # 保存五元组
            five_tup = (src_ip, dst_ip, src_port, dst_port, proto)
            five_tup_list.append(five_tup)

            if count == 0:
                access2fivetup_dict[access_RSID] = five_tup_list
                count += 1
            elif count != 0:
                if access_RSID in access2fivetup_dict:
                    access2fivetup_dict[access_RSID].append(five_tup)
                elif access_RSID not in access2fivetup_dict:
                    access2fivetup_dict[access_RSID] = five_tup_list
                count += 1
    # print(flow_dict[5739577766])#5739577766
    # print(flow_dict.keys())









