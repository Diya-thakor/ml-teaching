import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------------
# Streamlit setup
# ------------------------------
st.set_page_config(page_title="Multi-Channel Conv Visualizer", layout="wide")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'

st.title("Multi-Channel Conv2D Visualizer")
st.markdown("""
**One row per input channel**, output shown beside the **middle channel**.  
Each row: Input → Kernel → Product → Channel Sum →  
and the middle row shows the **final output matrix** building up one cell at a time.
""")

# ------------------------------
# Helpers
# ------------------------------
def get_output_size(n,k,s,p): return (n + 2*p - k)//s + 1
def apply_padding(m,p): return np.pad(m,p) if p>0 else m

@st.cache_data
def gen_data(h,w,c_in,c_out,k):
    np.random.seed(0)
    x  = np.random.randint(0,9,(c_in,h,w)).astype(float)
    kx = np.random.randn(c_out,c_in,k,k)
    return x,kx

# ------------------------------
# Visualization
# ------------------------------
def visualize_step(x,kx,stride,pad,c_out,step):
    C_in,H,W = x.shape
    C_out,_,K,_ = kx.shape
    colors = ['Reds','Greens','Blues']

    padded = np.stack([apply_padding(x[c],pad) for c in range(C_in)])
    Hp,Wp = padded.shape[1:]
    H_out = get_output_size(Hp,K,stride,0)
    W_out = get_output_size(Wp,K,stride,0)
    oi,oj = divmod(step,W_out)
    ii,jj = oi*stride, oj*stride
    kernels = kx[c_out]

    rf = padded[:,ii:ii+K,jj:jj+K]
    products = rf*kernels
    sums = products.reshape(C_in,-1).sum(axis=1)
    out_val = sums.sum()

    mid_row = C_in//2
    fig, axes = plt.subplots(C_in, 3, figsize=(14, 2.8*C_in))
    if C_in==1: axes = axes.reshape(1,4)

    # -------- Per-channel visualization --------
    for c in range(C_in):
        base_row = axes[c]

        # Input
        ax = base_row[0]
        ax.imshow(padded[c], cmap=colors[c%3], alpha=0.7)
        rect = patches.Rectangle((jj-0.5,ii-0.5),K,K,ec='red',lw=2,fc='none')
        ax.add_patch(rect)
        for i in range(Hp+1): ax.axhline(i-0.5,c='gray',lw=0.4)
        for j in range(Wp+1): ax.axvline(j-0.5,c='gray',lw=0.4)
        for i in range(Hp):
            for j in range(Wp):
                t=ax.text(j,i,f"{padded[c,i,j]:.0f}",ha='center',va='center',fontsize=8)
                t.set_bbox(dict(boxstyle='round,pad=0.2',fc='white',alpha=0.8,ec='none'))
        ax.set_title(f"In Ch {c}",fontsize=9)
        ax.axis('off')

        # Kernel
        ax = base_row[1]
        im=ax.imshow(kernels[c],cmap='RdBu_r',vmin=-1,vmax=1)
        for i in range(K+1):
            ax.axhline(i-0.5,c='gray',lw=0.4)
            ax.axvline(i-0.5,c='gray',lw=0.4)
        for i in range(K):
            for j in range(K):
                t=ax.text(j,i,f"{kernels[c,i,j]:.2f}",ha='center',va='center',fontsize=8)
                t.set_bbox(dict(boxstyle='round,pad=0.2',fc='white',alpha=0.8,ec='none'))
        ax.set_title(f"K{c}",fontsize=9)
        ax.axis('off')

        # Product
        ax = base_row[2]
        im=ax.imshow(products[c],cmap='Greens')
        for i in range(K+1):
            ax.axhline(i-0.5,c='gray',lw=0.4)
            ax.axvline(i-0.5,c='gray',lw=0.4)
        for i in range(K):
            for j in range(K):
                t=ax.text(j,i,f"{products[c,i,j]:.2f}",ha='center',va='center',fontsize=8)
                t.set_bbox(dict(boxstyle='round,pad=0.2',fc='white',alpha=0.8,ec='none'))
        ax.set_title(f"P{c} → Σ={sums[c]:.2f}",fontsize=9)
        ax.axis('off')

        # Channel scalar
        

    # -------- Output matrix (middle row) --------
    left = 0.95
    bottom = 0.12 + (mid_row)*(1/C_in)
    height = 1/C_in * 0.8
    width = 0.17
    ax_out = fig.add_axes([left, bottom, width, height])

    output = np.full((H_out,W_out),np.nan)
    for s in range(step+1):
        oi2,oj2 = divmod(s,W_out)
        ii2,jj2 = oi2*stride,oj2*stride
        rf2 = padded[:,ii2:ii2+K,jj2:jj2+K]
        output[oi2,oj2] = (rf2*kernels).sum()

    masked=np.ma.array(output,mask=np.isnan(output))
    im=ax_out.imshow(masked,cmap='viridis')
    for i in range(H_out+1): ax_out.axhline(i-0.5,c='gray',lw=0.5)
    for j in range(W_out+1): ax_out.axvline(j-0.5,c='gray',lw=0.5)
    for i in range(H_out):
        for j in range(W_out):
            if not np.isnan(output[i,j]):
                t=ax_out.text(j,i,f"{output[i,j]:.1f}",ha='center',va='center',fontsize=9)
                t.set_bbox(dict(boxstyle='round,pad=0.25',fc='white',ec='black',alpha=0.9,lw=0.8))
    rect=patches.Rectangle((oj-0.5,oi-0.5),1,1,ec='red',lw=2,fc='none')
    ax_out.add_patch(rect)
    ax_out.set_title(f"Output Step {step+1}",fontsize=9)
    ax_out.axis('off')

    expr = " + ".join([f"{s:.2f}" for s in sums]) + f" = **{out_val:.2f}**"
    st.markdown(f"### Output[{c_out},{oi},{oj}] = {expr}")

    plt.tight_layout()
    return fig

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Configuration")
H = st.sidebar.slider("Input Height",4,8,5)
W = st.sidebar.slider("Input Width",4,8,5)
C_in = st.sidebar.selectbox("Input Channels",[1,3],index=1)
C_out = st.sidebar.slider("Output Channels",1,3,1)
K = st.sidebar.slider("Kernel Size",2,4,3)
stride = st.sidebar.slider("Stride",1,2,1)
pad = st.sidebar.slider("Padding",0,2,0)

x,kx = gen_data(H,W,C_in,C_out,K)
Hp,Wp = H+2*pad, W+2*pad
H_out = get_output_size(Hp,K,stride,0)
W_out = get_output_size(Wp,K,stride,0)
steps = H_out*W_out
st.sidebar.markdown(f"**Output:** {C_out} × {H_out} × {W_out}")

c_out = st.sidebar.selectbox("Output Channel",list(range(C_out)))
step  = st.sidebar.slider("Step",0,steps-1,0)

# ------------------------------
# Main
# ------------------------------
st.header(f"Computing Output Channel {c_out} at Step {step+1}/{steps}")
fig = visualize_step(x,kx,stride,pad,c_out,step)
st.pyplot(fig)
plt.close('all')
