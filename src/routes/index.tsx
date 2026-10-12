import { createFileRoute } from "@tanstack/react-router";
import RossmannSimulator from "@/components/rossmann/RossmannSimulator";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Rossmann Sales Lab | Linear vs KNN" },
      { name: "description", content: "An interactive, browser-based Rossmann sales prediction lab comparing linear regression with K-nearest neighbors." },
      { property: "og:title", content: "Rossmann Sales Lab | Linear vs KNN" },
      { property: "og:description", content: "Explore, preprocess, train and compare Rossmann sales prediction models." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: RossmannSimulator,
});
