import asyncio
import statistics
import time
import httpx
from fastapi import Query
from findex.web.routes import app, lifespan, search


@app.get("/search_async")
async def search_async(q: str = Query("test"), k: int = Query(5)):
    return search(q=q, k=k, scorer="bm25", page=1)


async def run_benchmark(
    client: httpx.AsyncClient,
    endpoint: str,
    total_requests: int = 100,
    concurrency: int = 10,
):
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async def single_request():
        async with semaphore:
            t0 = time.perf_counter()
            resp = await client.get(endpoint, params={"q": "test", "k": 5})
            assert resp.status_code == 200
            latencies.append((time.perf_counter() - t0) * 1000)

    start_total = time.perf_counter()
    await asyncio.gather(*(single_request() for _ in range(total_requests)))
    elapsed_total = time.perf_counter() - start_total

    rps = total_requests / elapsed_total
    avg_ms = statistics.mean(latencies)
    p95_ms = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    return elapsed_total, rps, avg_ms, p95_ms


async def main():
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            await client.get("/search", params={"q": "test"})

            n_req = 100
            conc = 10
            sync_res = await run_benchmark(client, "/search", n_req, conc)
            async_res = await run_benchmark(client, "/search_async", n_req, conc)

            print(f"\nБенчмарк FastAPI ({n_req} запросов, concurrency={conc}):")
            print("-" * 68)
            print(f"{'Режим':<22} | {'Время (с)':<10} | {'RPS':<10} | {'Avg (мс)':<10} | {'P95 (мс)':<10}")
            print("-" * 68)
            print(f"{'def (ThreadPool)':<22} | {sync_res[0]:<10.4f} | {sync_res[1]:<10.1f} | {sync_res[2]:<10.2f} | {sync_res[3]:<10.2f}")
            print(f"{'async def (EventLoop)':<22} | {async_res[0]:<10.4f} | {async_res[1]:<10.1f} | {async_res[2]:<10.2f} | {async_res[3]:<10.2f}")
            print("-" * 68)


if __name__ == "__main__":
    asyncio.run(main())