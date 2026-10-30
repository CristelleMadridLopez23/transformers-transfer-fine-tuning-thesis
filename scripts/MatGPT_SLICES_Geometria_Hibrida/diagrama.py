"""Renderiza un esquema verificable de las dos ramas; no se requiere dataset."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def diagram(path):
    fig,ax=plt.subplots(figsize=(14,8.8));ax.set_xlim(0,14);ax.set_ylim(0,9);ax.axis('off')
    def box(x,y,w,h,txt,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',
             fc=color,ec='#52637C',lw=1.2))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=10,fontweight='semibold',linespacing=1.5)
    def line(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.5,color='#52637C'))
    fig.suptitle('Regresión energética híbrida Ridge–MLP8: SLICES y geometría de celda',
                 fontsize=17,fontweight='bold',y=.97)
    box(5.2,7.8,3.6,.65,'Tres SLICES ABS₃\n(misma composición)', '#DBEAFE')
    box(5.2,6.65,3.6,.65,'MatGPT pretrained · frozen\n12 bloques · hidden dim 768','#E9DFF7')
    line((7,7.8),(7,7.3))
    box(.8,5.35,4,.82,'L03_MEAN por estructura (768)\nPromedio de tres embeddings','#DBEAFE')
    box(5.1,5.35,3.8,.82,'FINAL_END por estructura\nEmbedding SLICES (768)','#DBEAFE')
    box(9.2,5.35,3.8,.82,'Geometría celda real\n7 descriptores (primitiva + Niggli)','#D9EFE5')
    line((7,6.65),(7,6.4));line((7,6.4),(2.8,6.4));line((2.8,6.4),(2.8,6.17))
    line((7,6.4),(7,6.17))
    box(.8,3.82,4,.85,'StandardScaler + Ridge histórico\nPredicción μc (eV/átomo) · fijo','#D9EFE5')
    box(5.1,3.82,7.9,.85,'Concat [FINAL_END, geometría train-scaled] → 775 dims\nMLP8: 775 → 8 (GELU + dropout 0.1) → 1 · 6,217 parámetros','#F7E7CE')
    line((2.8,5.35),(2.8,4.67));line((7,5.35),(7,4.67));line((11.1,5.35),(11.1,4.67))
    box(5.1,2.6,7.9,.65,'Centrado de las 3 salidas: δᵢ = rᵢ − media(rA,rB,rC)','#F7E7CE')
    line((9,3.82),(9,3.25))
    box(3.6,1.43,6.8,.72,'Predicción: eᵢ = μc + δᵢ (eV/átomo)','#DBEAFE')
    line((2.8,3.82),(2.8,1.79));line((2.8,1.79),(3.6,1.79));line((9,2.6),(9,2.15))
    box(2.6,.25,8.8,.66,'Derivación: E_total = N·eᵢ   |   ΔE = 1000(eB − eA)   |   GS = argmin(eA,eB,eC)','#D9EFE5')
    line((7,1.43),(7,.91))
    fig.text(.5,.08,'Control apareado: SLICES-only conserva 768→8→1 (6,161 parámetros). Ridge, folds y pérdidas son idénticos.',
             ha='center',fontsize=10,color='#52637C')
    fig.savefig(path,dpi=210,bbox_inches='tight',pad_inches=.3)
    plt.close(fig)
if __name__=='__main__':
    p=Path(__file__).resolve().parent/'diagrama_arquitectura_regresion_hibrida_con_geometria.png'
    diagram(p)
    print(p)
