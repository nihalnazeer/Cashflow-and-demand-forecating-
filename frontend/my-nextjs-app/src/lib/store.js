import { create } from "zustand";
import { persist } from "zustand/middleware";
import { 
  fetchHistoricalTrend, 
  fetchShopNames, 
  fetchKpis, 
  fetchPerformanceData 
} from './api';

// --- Mock Data Generators ---
// We'll keep these so your other components don't break
const generateForecastData = () => [
  { shop: "Downtown Store", item: "MacBook Pro 16\"", predictedDemand: 125, predictedSales: 187500, confidence: 89, trend: "up" },
  { shop: "Mall Location", item: "iPhone 15 Pro", predictedDemand: 98, predictedSales: 147000, confidence: 92, trend: "up" },
  { shop: "Online Store", item: "AirPods Pro", predictedDemand: 234, predictedSales: 175500, confidence: 85, trend: "up" },
];

export const useDashboardStore = create(
  persist(
    (set, get) => ({
      // --- State ---
      
      // ✨ State for mock data (for components not yet connected to an API)
      forecastData: generateForecastData(), 
      
      // State for filters
      filters: {
        timePeriod: "year_to_date",
        selectedShops: [],
      },
      // State for live data from our new APIs
      historicalTrend: [],
      isLoadingTrend: true,
      shopNames: [],
      kpis: { total_revenue: 0, total_demand: 0 },
      isLoadingKpis: true,
      performanceData: [],
      isLoadingPerformance: true,

      // --- Actions ---
      fetchHistoricalTrendData: async () => {
        set({ isLoadingTrend: true });
        const data = await fetchHistoricalTrend(get().filters);
        set({ historicalTrend: data, isLoadingTrend: false });
      },
      fetchShopList: async () => {
        const shops = await fetchShopNames();
        set({ shopNames: shops });
      },
      fetchKpiData: async () => {
        set({ isLoadingKpis: true });
        const data = await fetchKpis(get().filters);
        set({ kpis: data, isLoadingKpis: false });
      },
      fetchTopShopsData: async () => {
        set({ isLoadingPerformance: true });
        const data = await fetchPerformanceData(get().filters, 'shop_name', 'total_revenue');
        set({ performanceData: data, isLoadingPerformance: false });
      },

      setFiltersAndFetch: (newFilters) => {
        set((state) => ({ filters: { ...state.filters, ...newFilters } }));
        get().fetchHistoricalTrendData();
        get().fetchKpiData();
        get().fetchTopShopsData();
      },
      
      fetchAllInitialData: () => {
        get().fetchShopList();
        get().fetchHistoricalTrendData();
        get().fetchKpiData();
        get().fetchTopShopsData();
      },
    }),
    {
      name: "dashboard-storage",
      partialize: (state) => ({ filters: state.filters }),
    }
  )
);