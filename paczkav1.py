import streamlit as st
import plotly.graph_objects as go
import numpy as np
from itertools import permutations

# --- Packing logic (BLB) ---
class Product:
    def __init__(self, width, height, depth, name=""):
        self.original_dims = (width, height, depth)
        self.name = name

    def get_orientations(self):
        return set(permutations(self.original_dims))

class PackedProduct:
    def __init__(self, position, dimensions, name):
        self.position = position
        self.dimensions = dimensions
        self.name = name

def is_collision(pos, dims, placed):
    x1, y1, z1 = pos
    w1, d1, h1 = dims
    for p in placed:
        x2, y2, z2 = p.position
        w2, d2, h2 = p.dimensions
        if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + d2 and y1 + d1 > y2 and
            z1 < z2 + h2 and z1 + h1 > z2):
            return True
    return False

def find_best_position(product, placed, box_limit):
    best_pos = None
    best_dims = None
    best_score = None
    for dims in product.get_orientations():
        w,h,d = dims
        candidate_positions = [(0,0,0)]
        for p in placed:
            px, py, pz = p.position
            pw, pd, ph = p.dimensions
            candidate_positions.extend([
                (px+pw, py, pz),
                (px, py+pd, pz),
                (px, py, pz+ph)
            ])
        for pos in candidate_positions:
            x,y,z = pos
            if x+w <= box_limit[0] and y+d <= box_limit[1] and z+h <= box_limit[2]:
                if not is_collision(pos,(w,d,h),placed):
                    score = (x+y+z)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = pos
                        best_dims = (w,d,h)
    return best_pos, best_dims

def pack_products(products, box_limit):
    products_sorted = sorted(products, key=lambda p: (max(p.original_dims), np.prod(p.original_dims)), reverse=True)
    placed = []
    for p in products_sorted:
        pos,dims = find_best_position(p, placed, box_limit)
        if pos is None:
            return None, None
        placed.append(PackedProduct(pos,dims,p.name))
    max_x = max((p.position[0]+p.dimensions[0] for p in placed), default=0)
    max_y = max((p.position[1]+p.dimensions[1] for p in placed), default=0)
    max_z = max((p.position[2]+p.dimensions[2] for p in placed), default=0)
    return (max_x,max_y,max_z), placed

def cuboid_data(pos, size):
    x, y, z = pos
    dx, dy, dz = size
    return np.array([[x,y,z],[x+dx,y,z],[x+dx,y+dy,z],[x,y+dy,z],
                     [x,y,z+dz],[x+dx,y,z+dz],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]])

def cuboid_faces(verts):
    return [[verts[j] for j in [0,1,2,3]],
            [verts[j] for j in [4,5,6,7]],
            [verts[j] for j in [0,1,5,4]],
            [verts[j] for j in [2,3,7,6]],
            [verts[j] for j in [1,2,6,5]],
            [verts[j] for j in [4,7,3,0]]]

# --- Streamlit UI ---
st.set_page_config(page_title="PAKOWANIE Z MICHAŁEM", layout="wide")
st.title("PAKOWANIE Z MICHAŁEM")

if "products" not in st.session_state:
    st.session_state.products = []

# --- Funkcja ustawiająca wymiary pudełka na podstawie przewoźnika ---
def ustaw_wymiary_paczki(przewoznik):
    if przewoznik == "InPost Paczkomat":
        return "41 38 64"
    elif przewoznik == "Poczta Polska Kurier":
        return "65 42 40"
    elif przewoznik == "DPD Kurier":
        return "150 100 50"
    elif przewoznik == "Orlen Paczka":
        return "41 38 60"
    elif przewoznik == "Salon":
        return "32 34 64"
    else:
        return ""

# --- Layout: two columns ---
col1, col2 = st.columns([1,2])

# --- Left panel ---
with col1:
    st.markdown("""<div style="background-color:lightsteelblue; padding:10px; border-radius:5px; font-size:14px; max-height:600px; overflow-y:auto;">""", unsafe_allow_html=True)
    
    st.subheader("Wybierz przewoźnika")
    przewoznik = st.selectbox(
        "Wybierz przewoźnika:",
        ["", "InPost Paczkomat", "Poczta Polska Kurier", "DPD Kurier", "Orlen Paczka", "Salon"]
    )
    
    # Automatyczne ustawienie wymiarów pudełka
    domyslne_wymiary = ustaw_wymiary_paczki(przewoznik)
    boxdims_str = st.text_input("Wymiary pudełka (X Y Z):", domyslne_wymiary)

    st.subheader("Dodaj produkt")
    w = st.number_input("Szerokość", min_value=0.1, value=1.0)
    h = st.number_input("Wysokość", min_value=0.1, value=1.0)
    d = st.number_input("Głębokość", min_value=0.1, value=1.0)
    if st.button("Dodaj produkt"):
        name = f"P{len(st.session_state.products)+1}"
        st.session_state.products.append({"w":w,"h":h,"d":d,"name":name})

   st.subheader("Lista produktów")
for idx in range(len(st.session_state.products)):
    p = st.session_state.products[idx]
    colp1, colp2 = st.columns([4,1])
    with colp1:
        st.write(f"{p['name']}: {p['w']} x {p['h']} x {p['d']}")
    with colp2:
        if st.button("❌", key=f"remove_{p['name']}_{idx}"):
            st.session_state.products.pop(idx)
            break  # przerwij pętlę, Streamlit odświeży UI

st.markdown("</div>", unsafe_allow_html=True)
# --- Right panel: visualization 3D i podsumowanie ---
with col2:
    st.subheader("Wizualizacja pakowania")
    if st.button("Pakuj produkty"):
        if not st.session_state.products:
            st.error("Dodaj produkty przed pakowaniem!")
        else:
            try:
                box_limit = tuple(map(float, boxdims_str.strip().split()))
                if len(box_limit)!=3:
                    raise ValueError
            except:
                st.error("Nieprawidłowe wymiary pudełka!")
                box_limit=None

            if box_limit:
                product_objs = [Product(p['w'],p['h'],p['d'],p['name']) for p in st.session_state.products]
                box_size, layout = pack_products(product_objs, box_limit)
                if layout is None:
                    st.error("Nie udało się zmieścić produktów!")
                else:
                    fig = go.Figure()
                    # Pudełko
                    verts = cuboid_data((0,0,0), box_size)
                    faces = cuboid_faces(verts)
                    for face in faces:
                        x=[v[0] for v in face]+[face[0][0]]
                        y=[v[1] for v in face]+[face[0][1]]
                        z=[v[2] for v in face]+[face[0][2]]
                        fig.add_trace(go.Mesh3d(
                            x=x, y=y, z=z,
                            color='sandybrown', opacity=0.2,
                            i=[0,0,0,0], j=[1,2,3,4], k=[2,3,4,5],
                            name='Pudełko'
                        ))

                    colors=['red','blue','green','orange','purple','yellow','cyan','magenta']
                    for idx,p in enumerate(layout):
                        verts=cuboid_data(p.position,p.dimensions)
                        faces=cuboid_faces(verts)
                        for face in faces:
                            x=[v[0] for v in face]+[face[0][0]]
                            y=[v[1] for v in face]+[face[0][1]]
                            z=[v[2] for v in face]+[face[0][2]]
                            fig.add_trace(go.Scatter3d(
                                x=x, y=y, z=z,
                                mode='lines',
                                line=dict(color=colors[idx%len(colors)], width=5),
                                showlegend=False
                            ))
                        cx=p.position[0]+p.dimensions[0]/2
                        cy=p.position[1]+p.dimensions[1]/2
                        cz=p.position[2]+p.dimensions[2]/2
                        fig.add_trace(go.Scatter3d(
                            x=[cx], y=[cy], z=[cz],
                            text=[p.name],
                            mode='text',
                            showlegend=False
                        ))

                    fig.update_layout(scene=dict(
                        xaxis=dict(title='X', range=[0,box_size[0]]),
                        yaxis=dict(title='Y', range=[0,box_size[1]]),
                        zaxis=dict(title='Z', range=[0,box_size[2]]),
                        aspectmode='data'
                    ), margin=dict(l=0,r=0,b=0,t=0))

                    st.plotly_chart(fig, use_container_width=True)

                    # --- Podsumowanie ---
                    V_box = box_size[0]*box_size[1]*box_size[2]
                    V_products = sum(p.dimensions[0]*p.dimensions[1]*p.dimensions[2] for p in layout)
                    filled_percent = (V_products/V_box)*100
                    empty_percent = 100 - filled_percent

                    st.subheader("Podsumowanie")
                    st.text(f"Wymiary pudełka: {box_size[0]:.2f} x {box_size[1]:.2f} x {box_size[2]:.2f} cm")
                    st.text(f"Objętość pudełka: {V_box:.2f} cm³")
                    st.text(f"Objętość produktów: {V_products:.2f} cm³")
                    st.text(f"Wypełnienie: {filled_percent:.2f}%")
                    st.text(f"Pusta przestrzeń: {empty_percent:.2f}%")


