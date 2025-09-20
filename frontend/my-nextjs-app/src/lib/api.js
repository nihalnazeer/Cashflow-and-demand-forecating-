/**
 * Fetches historical trend data.
 * If filters are provided, it calls the '/detailed' endpoint.
 * Otherwise, it calls the '/overall' endpoint.
 * @param {object} filters - Optional filters like { timePeriod, selectedShops }
 */
export async function fetchHistoricalTrend(filters = {}) {
  try {
    const { timePeriod, selectedShops } = filters;
    let endpoint = "http://127.0.0.1:8000/analysis/historical/trends/overall";

    if (timePeriod || (selectedShops && selectedShops.length > 0)) {
      endpoint = "http://127.0.0.1:8000/analysis/historical/trends/detailed";
      const params = new URLSearchParams();

      if (timePeriod) {
        params.append('time_period', timePeriod);
      }
      if (selectedShops && selectedShops.length > 0) {
        selectedShops.forEach(shop => params.append('shop_name', shop));
      }
      endpoint += `?${params.toString()}`;
    }

    const response = await fetch(endpoint);
    if (!response.ok) throw new Error(`API call for trend failed! Status: ${response.status}`);
    
    const rawData = await response.json();
    return rawData.map(item => ({
      month: item.month,
      revenue: item.total_revenue,
    }));

  } catch (error) {
    console.error("❌ API Service: Error fetching trend data:", error);
    return [];
  }
}

/**
 * Fetches a unique list of all shop names from the backend.
 */
export async function fetchShopNames() {
  try {
    const response = await fetch("http://127.0.0.1:8000/analysis/historical/shops");
    if (!response.ok) throw new Error("API call for shops failed!");
    const data = await response.json();
    return data;
  } catch (error) { // ✨ FIX: Added curly braces around the catch block
    console.error("❌ API Service: Error fetching shop names:", error);
    return [];
  }
}

/**
 * Fetches aggregate KPIs (total revenue, total demand) for the given filters.
 * @param {object} filters - Optional filters like { timePeriod, selectedShops }
 */
export async function fetchKpis(filters = {}) {
  try {
    const { timePeriod, selectedShops } = filters;
    const params = new URLSearchParams();

    params.append('time_period', timePeriod || 'last_quarter');

    if (selectedShops && selectedShops.length > 0) {
      selectedShops.forEach(shop => params.append('shop_name', shop));
    }
    
    const endpoint = `http://127.0.0.1:8000/analysis/historical/kpis?${params.toString()}`;
    const response = await fetch(endpoint);
    if (!response.ok) throw new Error(`API call for KPIs failed! Status: ${response.status}`);
    
    return await response.json();
  } catch (error) {
    console.error("❌ API Service: Error fetching KPIs:", error);
    return { total_revenue: 0, total_demand: 0 }; // Return default object on error
  }
}

/**
 * Fetches dynamically grouped data (e.g., top shops by revenue).
 * @param {object} filters - Optional filters like { timePeriod, selectedShops }
 * @param {string} dimension - The dimension to group by (e.g., 'shop_name').
 * @param {string} metric - The metric to aggregate (e.g., 'total_revenue').
 */
export async function fetchPerformanceData(filters = {}, dimension = 'shop_name', metric = 'total_revenue') {
  try {
    const { timePeriod, selectedShops } = filters;
    const params = new URLSearchParams();

    params.append('metric', metric);
    params.append('time_period', timePeriod || 'last_quarter');

    if (selectedShops && selectedShops.length > 0) {
      selectedShops.forEach(shop => params.append('shop_name', shop));
    }

    const endpoint = `http://127.0.0.1:8000/analysis/historical/trends/${dimension}?${params.toString()}`;
    const response = await fetch(endpoint);
    if (!response.ok) throw new Error(`API call for performance data failed! Status: ${response.status}`);
    
    return await response.json();
  } catch (error) {
    console.error("❌ API Service: Error fetching performance data:", error);
    return [];
  }
}