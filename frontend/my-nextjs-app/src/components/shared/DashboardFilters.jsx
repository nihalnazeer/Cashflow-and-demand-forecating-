"use client";

import { useEffect } from "react";
import { useDashboardStore } from "../../lib/store";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/Select";
import { Button } from "../ui/Button";
import { Card, CardContent } from "../ui/Card";
import { CalendarIcon, BarChart3Icon, TargetIcon, StoreIcon, FilterIcon, SparklesIcon } from "lucide-react";

export default function DashboardFilters() {
  const { filters, setFiltersAndFetch, shopNames, fetchShopList } = useDashboardStore();

  useEffect(() => {
    fetchShopList();
  }, [fetchShopList]);

  // Handlers now call the same master action with the specific filter to change
  const handleFilterChange = (key, value) => {
    setFiltersAndFetch({ [key]: value });
  };

  const handleShopToggle = (shopName) => {
    const newShops = filters.selectedShops.includes(shopName)
      ? filters.selectedShops.filter(s => s !== shopName)
      : [...filters.selectedShops, shopName];
    setFiltersAndFetch({ selectedShops: newShops });
  };

  const FilterSection = ({ icon: Icon, label, children, className = "" }) => (
    <div className={`space-y-3 ${className}`}>
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 rounded-md bg-primary/10 border border-primary/20">
          <Icon className="h-3.5 w-3.5 text-primary" />
        </div>
        <label className="text-xs font-semibold text-foreground uppercase tracking-wider">
          {label}
        </label>
      </div>
      {children}
    </div>
  );

  const selectedShopsCount = filters.selectedShops?.length || 0;
  const totalShopsCount = shopNames.length;

  return (
    <Card className="animate-slide-in border-0 shadow-lg bg-gradient-to-r from-white to-gray-50/50">
      <CardContent className="p-6">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6 pb-4 border-b border-border/50">
          <div className="p-2 rounded-lg bg-gradient-to-br from-primary/20 to-primary/10 border border-primary/20">
            <FilterIcon className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Dashboard Filters</h3>
            <p className="text-xs text-muted-foreground">Customize your data view</p>
          </div>
        </div>

        {/* Filter Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-8">
          
          {/* Time Period */}
          <FilterSection icon={CalendarIcon} label="Time Period">
            <Select value={filters.timePeriod} onValueChange={(value) => handleFilterChange('timePeriod', value)}>
              <SelectTrigger className="h-10 bg-background/60 border-border/60 hover:border-primary/40 transition-colors">
                <SelectValue placeholder="Select period" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="last_7_days">Last 7 Days</SelectItem>
                <SelectItem value="last_30_days">Last 30 Days</SelectItem>
                <SelectItem value="last_90_days">Last 90 Days</SelectItem>
                <SelectItem value="last_6_months">Last 6 Months</SelectItem>
                <SelectItem value="last_year">Last Year</SelectItem>
              </SelectContent>
            </Select>
          </FilterSection>

          {/* Performance Dimension */}
          <FilterSection icon={BarChart3Icon} label="Analyze">
            <Select value={filters.performanceDimension} onValueChange={(value) => handleFilterChange('performanceDimension', value)}>
              <SelectTrigger className="h-10 bg-background/60 border-border/60 hover:border-primary/40 transition-colors">
                <SelectValue placeholder="Select dimension" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="shop_name">
                  <div className="flex items-center gap-2">
                    <StoreIcon className="h-4 w-4" />
                    Top Shops
                  </div>
                </SelectItem>
                <SelectItem value="item_category_name">
                  <div className="flex items-center gap-2">
                    <SparklesIcon className="h-4 w-4" />
                    Top Categories
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </FilterSection>

          {/* Performance Metric */}
          <FilterSection icon={TargetIcon} label="By Metric">
            <Select value={filters.performanceMetric} onValueChange={(value) => handleFilterChange('performanceMetric', value)}>
              <SelectTrigger className="h-10 bg-background/60 border-border/60 hover:border-primary/40 transition-colors">
                <SelectValue placeholder="Select metric" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="total_revenue">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    Total Revenue
                  </div>
                </SelectItem>
                <SelectItem value="total_demand">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                    Total Demand
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </FilterSection>

          {/* Shop Selection */}
          <FilterSection 
            icon={StoreIcon} 
            label={`Shop Filter (${selectedShopsCount}/${totalShopsCount})`}
            className="md:col-span-2 lg:col-span-1 xl:col-span-2"
          >
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                {shopNames.slice(0, 6).map(shop => (
                  <Button
                    key={shop}
                    variant={filters.selectedShops?.includes(shop) ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleShopToggle(shop)}
                    className={`h-8 px-3 text-xs font-medium transition-all duration-200 ${
                      filters.selectedShops?.includes(shop) 
                        ? 'shadow-md bg-primary hover:bg-primary/90 border-primary' 
                        : 'hover:bg-primary/10 hover:border-primary/40 hover:text-primary'
                    }`}
                  >
                    <span className="truncate max-w-[80px]">{shop}</span>
                    {filters.selectedShops?.includes(shop) && (
                      <div className="ml-1 w-1.5 h-1.5 rounded-full bg-primary-foreground"></div>
                    )}
                  </Button>
                ))}
              </div>
              
              {shopNames.length > 6 && (
                <div className="text-xs text-muted-foreground border-t border-border/30 pt-2">
                  {shopNames.length - 6} more shops available
                </div>
              )}
              
              {selectedShopsCount > 0 && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">
                    {selectedShopsCount} shop{selectedShopsCount !== 1 ? 's' : ''} selected
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setFiltersAndFetch({ selectedShops: [] })}
                    className="h-6 px-2 text-xs text-muted-foreground hover:text-foreground"
                  >
                    Clear all
                  </Button>
                </div>
              )}
            </div>
          </FilterSection>
          
        </div>
      </CardContent>
    </Card>
  );
}