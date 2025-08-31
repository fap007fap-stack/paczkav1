with col1:
    # scrollable wstążka bez niebieskiego tła pod nagłówkiem
    st.markdown("""
    <div style="background-color:lightsteelblue; padding:10px; border-radius:5px; font-size:12px; max-height:600px; overflow-y:auto;">
    """, unsafe_allow_html=True)
    
    st.subheader("Dodaj produkt")
    w = st.number_input("Szerokość", min_value=0.1, value=1.0)
    h = st.number_input("Wysokość", min_value=0.1, value=1.0)
    d = st.number_input("Głębokość", min_value=0.1, value=1.0)
    if st.button("Dodaj produkt"):
        name = f"P{len(st.session_state.products)+1}"
        st.session_state.products.append({"w":w,"h":h,"d":d,"name":name})

    st.subheader("Lista produktów")
    for i,p in enumerate(st.session_state.products):
        colp1, colp2 = st.columns([4,1])
        with colp1:
            st.write(f"{p['name']}: {p['w']} x {p['h']} x {p['d']}")
        with colp2:
            if st.button("❌", key=f"del_{i}"):
                st.session_state.products.pop(i)
                st.experimental_rerun()

    st.subheader("Wymiary pudełka (X Y Z)")
    boxdims_str = st.text_input("Np. 30 20 10", "30 20 10")

    st.markdown("</div>", unsafe_allow_html=True)
