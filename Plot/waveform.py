import numpy as np
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):
        pass

    def plot1D_line(self, x):
        # 绘制一条线
        # 去除异常值
        plt.figure()
        plt.plot(x, color = 'b')
        plt.show()

    def plot1D_line_group(self, x, row_n=2, col_n =4):
        # 每次绘制一条线, 输入[T, C], 共C个子图
        plt.figure(figsize=(16, 8))  

        # 绘制row_n  x  col_n个子图
        for i in range(x.shape[1]):
            plt.subplot(row_n, col_n, i + 1)  
            plt.plot(x[:, i], color='b')  
            plt.title(f'The raw signal of {i + 1}th channel sEMG')
            plt.xlabel('Sample point')
            plt.ylabel('Amplitude')

        plt.tight_layout()  # 调整子图间的间距和边缘
        plt.show()
    
    def plot2D_line_group(self, x1, x2, x3, row_n=2, col_n =4):
        # 绘制row_n  x  col_n个子图
        x = [i for i in range(1, x1.shape[0]+1)]
        print(x1.shape)
        for i in range(x1.shape[1]):
            plt.subplot(row_n, col_n, i + 1)  
            plt.plot(x, x1[:, i], color='b')  
            plt.plot(x, x2[:, i], color='r')  
            plt.plot(x, x3[:, i], color='g')  


            plt.title(f'The raw signal of {i + 1}th channel sEMG')
            plt.xlabel('Sample point')
            plt.ylabel('Amplitude')


    def plot2D_line(self, x1, x2, x3):
        x = [i for i in range(1, x1.shape[0]+1)]

        # 绘制多条线
        plt.plot(x, x1, color='b')  
        plt.plot(x, x2, color='r')  
        plt.plot(x, x3, color='g')  

        plt.title(f'The raw signal of channel sEMG')
        plt.xlabel('Sample point')
        plt.ylabel('Amplitude')

        plt.show()

    def plot_scatter(self, x, y, title, scale = False, xlim=None, ylim=None):
        # 绘制散点图
        plt.figure()
        plt.plot(x, y, color='b')
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        if scale:
            # Define the zoomed-in area
            zoom_region_x = [0.01, 6]
            zoom_region_y = [-1, 8e07]
            # Add a zoomed-in subplot
            ax_zoom = plt.axes([0.25, 0.5, 0.6, 0.3])  # Adjust position and size as needed
            ax_zoom.plot(x, y, color='b')
            ax_zoom.set_xlim(zoom_region_x)
            ax_zoom.set_ylim(zoom_region_y)
            ax_zoom.legend()
        plt.show()

