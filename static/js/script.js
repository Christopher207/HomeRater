/**
 * Mapa interactivo de inmuebles con filtros y sidebar
 * Compatible con main.js existente
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Inicializando mapa de inmuebles...');
    
    // 1. Verificar si estamos en la página correcta
    const mapElement = document.getElementById('map');
    if (!mapElement) {
        console.log('No se encontró el elemento #map - Saliendo de script.js');
        return;
    }

    // 2. Inicialización del mapa con control de errores
    var map;
    try {
        /**map = L.map('map', {
            *center: [-12.0464, -77.0428], // Coordenadas de Lima
            *zoom: 12,
            *zoomControl: true,
            *preferCanvas: true // Mejor rendimiento para muchos marcadores
        });*/
	map = L.map('map').setView([-12.0464, -77.0428],17);
        console.log('Mapa Leaflet inicializado correctamente');
    } catch (error) {
        console.error('Error al inicializar el mapa:', error);
        return;
    }

    // 3. Configuración de capas base
    const baseLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
        detectRetina: true
    }).addTo(map);

    // 4. Grupo de clusters para marcadores
    const markers = L.markerClusterGroup({
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        zoomToBoundsOnClick: true,
        disableClusteringAtZoom: 16,
        chunkedLoading: true // Mejor rendimiento para muchos marcadores
    });

    // 5. Variables de estado
    let allProperties = [];
    let selectedProperties = [];
    let currentFilters = {
        propertyType: 'all',
        contractType: 'all'
    };

    // 6. Cargar datos de inmuebles
    function loadPropertyData() {
        console.log('Cargando datos de inmuebles...');
        
        fetch('/api/properties')
            .then(response => {
                if (!response.ok) throw new Error('Error en la respuesta del servidor');
                return response.json();
            })
            .then(data => {
                allProperties = data;
                console.log(`Datos cargados: ${data.length} inmuebles`);
                updateMapMarkers();
                setupEventListeners();
                
                // Forzar redibujado del mapa si es necesario
                setTimeout(() => map.invalidateSize(), 100);
            })
            .catch(error => {
                console.error('Error al cargar datos:', error);
                showError('No se pudieron cargar los datos. Por favor recarga la página.');
            });
    }

    // 7. Actualizar marcadores en el mapa según filtros
    function updateMapMarkers() {
        console.log('Actualizando marcadores con filtros:', currentFilters);
        
        // Filtrar propiedades
        const filtered = allProperties.filter(property => {
            const typeMatch = currentFilters.propertyType === 'all' || 
                            property.tipo.toLowerCase() === currentFilters.propertyType;
            const contractMatch = currentFilters.contractType === 'all' || 
                               property.contrato.toLowerCase() === currentFilters.contractType;
            return typeMatch && contractMatch;
        });

        // Limpiar marcadores existentes
        markers.clearLayers();
        
        // Añadir nuevos marcadores
        filtered.forEach(property => {
            const marker = L.marker(property.coords, {
                propertyId: property.id,
                riseOnHover: true
            }).bindPopup(
                `<b>${property.titulo}</b><br>
                <small>${property.precio}</small><br>
                <button class="map-popup-btn" data-id="${property.id}">Ver detalles</button>`,
                { maxWidth: 250 }
            );
            
            marker.on('click', function() {
                addToSelectedProperties(property);
            });
            
            markers.addLayer(marker);
        });

        // Añadir cluster al mapa
        map.addLayer(markers);
        
        // Ajustar vista si hay marcadores
        if (filtered.length > 0) {
            map.fitBounds(markers.getBounds(), { padding: [50, 50] });
        }
    }

    // 8. Manejar propiedades seleccionadas
    function addToSelectedProperties(property) {
        if (!selectedProperties.some(p => p.id === property.id)) {
            selectedProperties.push(property);
            renderSelectedProperties();
            
            // Animación con ScrollReveal si está disponible
            if (window.sr) {
                sr.reveal('.property-card:last-child', {
                    duration: 400,
                    distance: '20px',
                    easing: 'cubic-bezier(0.5, -0.01, 0, 1.005)',
                    origin: 'right'
                });
            }
        }
    }

    function removeFromSelectedProperties(propertyId) {
        selectedProperties = selectedProperties.filter(p => p.id !== propertyId);
        renderSelectedProperties();
    }

    function renderSelectedProperties() {
        const container = document.getElementById('selected-properties');
        if (!container) return;
        
        container.innerHTML = selectedProperties.length === 0 
            ? '<p class="no-properties">Selecciona inmuebles en el mapa</p>'
            : selectedProperties.map(property => `
                <div class="property-card" data-id="${property.id}">
                    <button class="close-btn" aria-label="Remover propiedad">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z"/>
                        </svg>
                    </button>
                    <h4>${property.titulo}</h4>
                    <div class="property-meta">
                        <span class="property-type">${property.tipo}</span>
                        <span class="property-contract">${property.contrato}</span>
                    </div>
                    <p class="property-price">${property.precio}</p>
                    <p class="property-location">${property.ubicacion}</p>
                    <p class="property-desc">${property.descripcion.substring(0, 120)}...</p>
                    <img src="${property.imagen}" alt="${property.titulo}" loading="lazy" class="property-image">
                    <button class="view-on-map-btn" data-coords="${property.coords}">Ver en mapa</button>
                </div>
            `).join('');
        
        // Event listeners para los nuevos elementos
        container.querySelectorAll('.close-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const card = e.target.closest('.property-card');
                if (card) {
                    removeFromSelectedProperties(card.dataset.id);
                }
            });
        });
        
        container.querySelectorAll('.view-on-map-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const coords = JSON.parse(e.target.dataset.coords);
                map.setView(coords, 16, { animate: true });
            });
        });
    }

    // 9. Configurar controles de filtro
    function setupEventListeners() {
        // Filtros
        document.getElementById('property-type')?.addEventListener('change', (e) => {
            currentFilters.propertyType = e.target.value;
            updateMapMarkers();
        });
        
        document.getElementById('contract-type')?.addEventListener('change', (e) => {
            currentFilters.contractType = e.target.value;
            updateMapMarkers();
        });
        
        document.getElementById('apply-filters')?.addEventListener('click', () => {
            updateMapMarkers();
        });
        
        // Redimensionamiento del mapa
        window.addEventListener('resize', () => {
            map.invalidateSize();
        });
    }

    // 10. Mostrar errores al usuario
    function showError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'map-error-message';
        errorElement.textContent = message;
        document.querySelector('.map-container')?.prepend(errorElement);
        
        setTimeout(() => {
            errorElement.remove();
        }, 5000);
    }

    // 11. Inicialización completa
    loadPropertyData();
    
    // Debug: Exponer variables para inspección
    window.mapDebug = {
        map,
        markers,
        allProperties,
        selectedProperties,
        currentFilters
    };
});