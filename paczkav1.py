import streamlit as st
import plotly.graph_objects as go
import numpy as np
from itertools import permutations
import random

# === Packing logic ===

class Product:
    def __init__(self, width, height, depth, name=""):
        self.original_dims = (width, height, depth)
        self.name = name

    def get_orientations(self):
        # wszystkie 6 permutacji wymiarów
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
    best_score = None  # minimalna objętość pustej przestrzeni wokół

    for dims in product.get_orientations():
        w, h, d = dims
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
            x, y, z = pos
            if x + w <= box_limit[0] and y + d <= box_limit[1] and z + h <= box_limit[2]:
                if not is_collision(pos, (w,d,h), placed):
                    # Heurystyka: minimalizujemy objętość pudełka do tej pozycji
                    score = (x+w)*(y+d)*(z+h)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = pos
                        best_dims = (w,d,h)
    return best_pos, best_dims

def pack_products(products, box_limit):
    # sortowanie po objętości malejąco
    products_sorted = sorted(products, key=lambda p: p.original_dims[0]*p.original_dims[1]*p.original_dims[2], reverse=True)
    placed = []
    max_x = max_y = max_z = 0
    for p in products_sorted:
        pos, dims = find_best_position(p, placed, box_limit)
        if pos is None:
            return None, None
        placed.append(PackedProduct(pos, dims, p.name))
        max_x = max(max_x, pos[0]+dims[0])
        max_y = max(max_y, pos[1]+dims[1])
        max_z = max(max_z, pos[2]+dims[2])
    return (max_x, max_y, max_z), placed

def cuboid_data(pos, size):
    x, y, z = pos
    dx, dy, dz = size
    return np.array([
        [x, y, z],
        [x+dx, y, z],
        [x+dx, y+dy, z],
        [x, y+dy, z],
        [x, y, z+dz],
        [x+dx, y, z+dz],
        [x+dx, y+dy, z+dz],
        [x, y+dy, z+dz]
    ])

def cuboid_faces(verts):
    return [
        [verts[j] for j in [0,1,2,3]],
        [verts[j] for j in [4,5,6,7]],
        [verts[j] for j in [0,1,5,4]],
        [verts[j] for j in [2,3,7,6]],
        [verts[j] for j in [1,2,6,5]],
        [verts[j] for j in [4,7,3,0]]
    ]

# === Streamlit UI ===
st.set_page_config(page_title="3D Packing Online", layout="wide")
st.title("Pakowanie produktów 3D (Online)")

if "products" not in st.session_state:
    st.session_state.products = []

# --- Sidebar controls ---
with st.sidebar:
    st.header("Dodaj produkt")
    w = st.number_input("Szerokość", min_value=0.1, value=1.0)
    h = st.number_input("Wysokość", min_value=0.1, value=1.0)
    d = st.number_input("Głębokość", min_value=0.1, value=1.0)
    if st.button("Dodaj produkt"):
        name = f"P{len(st.session_state.products)+1}"
        st.session_state.products.append({"w": w, "h": h, "d": d, "name": name})

    if st.button("Resetuj listę"):
        st.session_state.products = []

    st.header("Lista produktów")
    for p in st.session_state.products:
        st.write(f"{p['name']}: {p['w']} x {p['h']} x {p['d']}")

    st.header("Wymiary pudełka (X Y Z)")
    boxdims_str = st.text_input("Np. 30 20 10", "30 20 10")

# --- Main panel: packing results ---
st.subheader("Wynik pakowania")

if st.button("Pakuj produkty"):
    if not st.session_state.products:
        st.error("Dodaj produkty przed pakowaniem!")
    else:
        try:
            box_limit = tuple(map(float, boxdims_str.strip().split()))
            if len(box_limit) != 3:
                raise ValueError
        except:
            st.error("Nieprawidłowe wymiary pudełka!")
            box_limit = None

        if box_limit:
            product_objs = [Product(p['w'], p['h'], p['d'], p['name']) for p in st.session_state.products]
            box_size, layout = pack_products(product_objs, box_limit)

            if layout is None:
                st.error("Nie udało się zmieścić produktów w zadanym pudełku!")
            else:
                # 3D plot
                fig = go.Figure()
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
                for idx, p in enumerate(layout):
                    verts = cuboid_data(p.position, p.dimensions)
                    faces = cuboid_faces(verts)
                    for face in faces:
                        x = [vertex[0] for vertex in face] + [face[0][0]]
                        y = [vertex[1] for vertex in face] + [face[0][1]]
                        z = [vertex[2] for vertex in face] + [face[0][2]]
                        fig.add_trace(go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='lines',
                            line=dict(color=colors[idx % len(colors)], width=5),
                            showlegend=False
                        ))
                    cx = p.position[0] + p.dimensions[0]/2
                    cy = p.position[1] + p.dimensions[1]/2
                    cz = p.position[2] + p.dimensions[2]/2
                    fig.add_trace(go.Scatter3d(
                        x=[cx], y=[cy], z=[cz],
                        text=[p.name],
                        mode='text',
                        showlegend=False
                    ))

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title="X [cm]", range=[0, box_size[0]]),
                        yaxis=dict(title="Y [cm]", range=[0, box_size[1]]),
                        zaxis=dict(title="Z [cm]", range=[0, box_size[2]])
                    ),
                    margin=dict(l=0,r=0,b=0,t=0)
                )

                st.plotly_chart(fig, use_container_width=True)

                # summary
                V_box = box_size[0]*box_size[1]*box_size[2]
                V_products = sum(p.dimensions[0]*p.dimensions[1]*p.dimensions[2] for p in layout)
                filled_percent = (V_products/V_box)*100
                empty_percent = 100 - filled_percent

                st.subheader("Podsumowanie")
                st.text(f"Pudełko: {box_size[0]:.2f} x {box_size[1]:.2f} x {box_size[2]:.2f} cm")
                st.text(f"Objętość pudełka: {V_box:.2f} cm³")
                st.text(f"Objętość produktów: {V_products:.2f} cm³")
                st.text(f"Zajętość: {filled_percent:.2f}%")
                st.text(f"Pusta przestrzeń: {empty_percent:.2f}%")
