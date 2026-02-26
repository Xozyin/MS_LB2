from classes.formPainter import FormPainter
from classes.globalConfig import GlobalConfig

def main():
    x0 = [0.0, 0.0, 1100.0]
    T_interval = (0.0, 11.0)
    c = 8000
    u = 20
    h_tv = 9900
    g = 9.81

    globConf = GlobalConfig(x0=x0, T_interval=T_interval, c=c, u=u, h_tv=h_tv, g=g)
    fp = FormPainter(globConf) 

main()