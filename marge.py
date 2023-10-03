import wave
import numpy as np

#用于读取wav文件的数据和参数
def read_wav(filename):
    # 打开wav文件
    wav = wave.open(filename, "rb")
    # 获取wav文件的参数
    params = wav.getparams()
    # 获取wav文件的声道数，采样宽度，采样率和帧数
    nchannels, sampwidth, framerate, nframes = params[:4]
    # 读取wav文件的数据
    data = wav.readframes(nframes)
    # 关闭wav文件
    wav.close()
    # 将数据转换为numpy数组
    data = np.frombuffer(data, dtype=np.int16)
    # 返回数据和参数
    return data, params

# 用于写入wav文件的数据和参数
def write_wav(filename, data, params):
    # 打开wav文件
    wav = wave.open(filename, "wb")
    # 设置wav文件的参数
    wav.setparams(params)
    # 将数据转换为字节流
    data = data.tobytes()
    # 写入wav文件的数据
    wav.writeframes(data)
    # 关闭wav文件
    wav.close()


rate=44100

for i in range(1, 36):
    # 定义两个wav文件的路径
    file1 = "./samples/Ejafjalla/Ejafjalla-"+str(i)+".wav"
    for j in range(1, 36):
        file2 = "./samples/Swire/Swire-"+str(j)+".wav"

        # 读取两个wav文件的数据和参数
        data1, params1 = read_wav(file1)
        data2, params2 = read_wav(file2)

        # print(params1)
        # print(params2)

        # 判断两个wav文件是否有相同的声道数，采样宽度和采样率
        params1 = params1._replace(framerate=rate)
        params2 = params2._replace(framerate=rate)

        # 获取两个wav文件的声道数，采样宽度，采样率和帧数
        nchannels1, sampwidth1, framerate1, nframes1 = params1[:4]
        nchannels2, sampwidth2, framerate2, nframes2 = params2[:4]

        if nchannels1 == nchannels2 and sampwidth1 == sampwidth2:
            # 如果有相同的声道数，采样宽度和采样率，那么可以将两个wav文件合并同时播放
            # 获取两个wav文件中较长的帧数
            max_nframes = rate * 4
            # 如果第一个wav文件的帧数小于较长的帧数，那么需要补零
            if nframes1 < max_nframes:
                # 计算需要补零的长度
                pad_len = max_nframes - nframes1
                # 创建一个全零的数组，长度为补零的长度乘以声道数
                pad_data = np.zeros(pad_len * nchannels1, dtype=np.int16)
                # 将原始数据和补零的数组拼接起来，形成新的数据
                data1 = np.concatenate((data1, pad_data))
                # 更新第一个wav文件的帧数为较长的帧数
                nframes1 = max_nframes
                # 更新第一个wav文件的参数中的帧数为较长的帧数
                params1 = params1._replace(nframes=nframes1)
            # 如果第二个wav文件的帧数小于较长的帧数，那么需要补零
            if nframes2 < max_nframes:
                # 计算需要补零的长度
                pad_len = max_nframes - nframes2
                # 创建一个全零的数组，长度为补零的长度乘以声道数
                pad_data = np.zeros(pad_len * nchannels2, dtype=np.int16)
                # 将原始数据和补零的数组拼接起来，形成新的数据
                data2 = np.concatenate((data2, pad_data))
                # 更新第二个wav文件的帧数为较长的帧数
                nframes2 = max_nframes
                # 更新第二个wav文件的参数中的帧数为较长的帧数
                params2 = params2._replace(nframes=nframes2)

            if nframes1 > max_nframes:
                pad_len = nframes1 - max_nframes
                data1 = data1[:-pad_len]
                # 更新第二个wav文件的帧数为较长的帧数
                nframes1 = max_nframes
                # 更新第二个wav文件的参数中的帧数为较长的帧数
                params1 = params1._replace(nframes=nframes1)

            if nframes2 > max_nframes:
                pad_len = nframes2 - max_nframes
                data2 = data2[:-pad_len]
                # 更新第二个wav文件的帧数为较长的帧数
                nframes2 = max_nframes
                # 更新第二个wav文件的参数中的帧数为较长的帧数
                params2 = params2._replace(nframes=nframes2)

            # 将两个wav文件的数据相加，形成合并的数据
            if params1 == params2:
                data = data1 + data2
                # 定义一个新的wav文件的路径
                file1 = "./samples/1_Split/1_Split-"+str(35*(i-1)+j)+".wav"
                file2 = "./samples/2_Split/2_Split-"+str(35*(i-1)+j)+".wav"
                file3 = "./samples/Merge/Merge-"+str(35*(i-1)+j)+".wav"
                # 写入新的wav文件的数据和参数
                write_wav(file1, data1, params1)
                write_wav(file2, data2, params2)
                write_wav(file3, data, params1)
                # 打印成功的信息
                print(str(35*(i-1)+j)+" Succeed")
            else:
                print(str(35*(i-1)+j)+" Failed")
        else:
            # 如果没有相同的声道数，采样宽度和采样率，那么无法将两个wav文件合并同时播放
            # 打印失败的信息
            print(str(35*(i-1)+j)+" Failed")
